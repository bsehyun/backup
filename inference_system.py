import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
import joblib
import os
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceSystem:
    """
    CSV 파일과 목표 시간을 입력받아 추론을 수행하고 결과를 CSV에 추가하는 시스템
    """
    
    def __init__(self, model_dir: str = "./models"):
        """
        초기화
        
        Args:
            model_dir: 모델과 스케일러가 저장된 디렉토리 경로
        """
        self.model_dir = model_dir
        self.short_model = None
        self.long_model = None
        self.short_scaler = None
        self.long_scaler = None
        
        # 모델과 스케일러 로드
        self._load_models()
        
        # 필요한 최소 데이터 길이 (분 단위)
        self.min_data_length = 60  # 1시간
        
    def _load_models(self):
        """모델과 스케일러를 로드합니다."""
        try:
            model_files = {
                'short_model': 'short_model.pkl',
                'long_model': 'long_model.pkl', 
                'short_scaler': 'short_scaler.pkl',
                'long_scaler': 'long_scaler.pkl'
            }
            
            for attr_name, filename in model_files.items():
                filepath = os.path.join(self.model_dir, filename)
                if os.path.exists(filepath):
                    setattr(self, attr_name, joblib.load(filepath))
                    logger.info(f"로드됨: {filename}")
                else:
                    logger.warning(f"파일을 찾을 수 없음: {filepath}")
                    
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            raise
    
    def _validate_input_csv(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        입력 CSV의 유효성을 검사합니다.
        
        Args:
            df: 입력 데이터프레임
            
        Returns:
            (유효성 여부, 오류 메시지)
        """
        # 기본 검사
        if df.empty:
            return False, "입력 CSV가 비어있습니다."
        
        if df.index.name is None:
            return False, "CSV의 인덱스가 시간으로 설정되어 있지 않습니다."
        
        # 시간 인덱스 검사
        try:
            pd.to_datetime(df.index)
        except:
            return False, "인덱스가 유효한 시간 형식이 아닙니다."
        
        # 데이터 길이 검사
        if len(df) < self.min_data_length:
            return False, f"데이터가 충분하지 않습니다. 최소 {self.min_data_length}개 행이 필요합니다."
        
        # 시간 순서 검사
        if not df.index.is_monotonic_increasing:
            return False, "시간이 오름차순으로 정렬되어 있지 않습니다."
        
        return True, ""
    
    def _get_required_features(self) -> Dict[str, List[str]]:
        """
        각 모델에 필요한 피처 목록을 반환합니다.
        실제 구현에서는 모델의 피처 이름을 분석하여 동적으로 생성해야 합니다.
        """
        # 예시 피처 목록 (실제로는 모델의 피처 이름을 분석하여 생성)
        return {
            'short_model': ['feature1', 'feature2', 'feature3'],
            'long_model': ['feature1', 'feature2', 'feature4', 'feature5']
        }
    
    def _validate_features(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        필요한 피처들이 CSV에 있는지 검사합니다.
        
        Args:
            df: 입력 데이터프레임
            
        Returns:
            (유효성 여부, 오류 메시지)
        """
        required_features = self._get_required_features()
        all_required = set()
        
        for model_features in required_features.values():
            all_required.update(model_features)
        
        missing_features = all_required - set(df.columns)
        
        if missing_features:
            return False, f"필요한 피처가 누락되었습니다: {list(missing_features)}"
        
        return True, ""
    
    def _create_lagged_features(self, df: pd.DataFrame, feature_name: str, 
                               lag: int, time_minutes: int) -> pd.Series:
        """
        지연 피처를 생성합니다.
        
        Args:
            df: 입력 데이터프레임
            feature_name: 원천 피처 이름
            lag: 지연 값
            time_minutes: 시간(분)
            
        Returns:
            생성된 지연 피처
        """
        if feature_name not in df.columns:
            raise ValueError(f"피처 {feature_name}이 데이터에 없습니다.")
        
        # 시간 간격을 고려한 지연 계산
        time_interval = pd.Timedelta(minutes=time_minutes)
        lagged_data = df[feature_name].shift(lag)
        
        return lagged_data
    
    def _create_roc_features(self, df: pd.DataFrame, feature_name: str,
                           roc_period: int, time_minutes: int, 
                           agg_type: str) -> pd.Series:
        """
        변화율 피처를 생성합니다.
        
        Args:
            df: 입력 데이터프레임
            feature_name: 원천 피처 이름
            roc_period: 변화율 계산 기간
            time_minutes: 시간(분)
            agg_type: 집계 타입 ('median' 또는 'std')
            
        Returns:
            생성된 변화율 피처
        """
        if feature_name not in df.columns:
            raise ValueError(f"피처 {feature_name}이 데이터에 없습니다.")
        
        # 변화율 계산
        roc = df[feature_name].pct_change(roc_period)
        
        # 이동 집계
        if agg_type == 'median':
            result = roc.rolling(window=time_minutes).median()
        elif agg_type == 'std':
            result = roc.rolling(window=time_minutes).std()
        else:
            raise ValueError(f"지원하지 않는 집계 타입: {agg_type}")
        
        return result
    
    def _extract_feature_name_info(self, feature_name: str) -> Dict:
        """
        피처 이름에서 변형 정보를 추출합니다.
        
        Args:
            feature_name: 피처 이름
            
        Returns:
            변형 정보 딕셔너리
        """
        parts = feature_name.split('_')
        
        # 원천 피처만 있는 경우
        if len(parts) == 1:
            return {
                'base_feature': parts[0],
                'type': 'original'
            }
        
        # 지연 피처인 경우: {원천feature이름}_{lag}_{시간}m
        if len(parts) >= 3 and parts[-1].endswith('m'):
            try:
                lag = int(parts[-2])
                time_minutes = int(parts[-1][:-1])  # 'm' 제거
                base_feature = '_'.join(parts[:-2])
                return {
                    'base_feature': base_feature,
                    'type': 'lag',
                    'lag': lag,
                    'time_minutes': time_minutes
                }
            except ValueError:
                pass
        
        # 변화율 피처인 경우: {원천feature이름}_{roc}_{시간}m_{median or std}
        if len(parts) >= 4 and parts[1] == 'roc' and parts[-2].endswith('m'):
            try:
                roc_period = int(parts[2])
                time_minutes = int(parts[-2][:-1])  # 'm' 제거
                agg_type = parts[-1]
                base_feature = parts[0]
                
                if agg_type in ['median', 'std']:
                    return {
                        'base_feature': base_feature,
                        'type': 'roc',
                        'roc_period': roc_period,
                        'time_minutes': time_minutes,
                        'agg_type': agg_type
                    }
            except ValueError:
                pass
        
        # 파싱할 수 없는 경우
        return {
            'base_feature': feature_name,
            'type': 'unknown'
        }
    
    def _prepare_features(self, df: pd.DataFrame, model_type: str) -> pd.DataFrame:
        """
        모델에 필요한 피처를 준비합니다.
        
        Args:
            df: 입력 데이터프레임
            model_type: 모델 타입 ('short' 또는 'long')
            
        Returns:
            준비된 피처 데이터프레임
        """
        if model_type == 'short':
            model = self.short_model
            scaler = self.short_scaler
        else:
            model = self.long_model
            scaler = self.long_scaler
        
        if model is None:
            raise ValueError(f"{model_type}_model이 로드되지 않았습니다.")
        
        # 모델의 피처 이름 가져오기
        feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else []
        
        if not feature_names:
            raise ValueError(f"{model_type}_model의 피처 이름을 가져올 수 없습니다.")
        
        # 피처 준비
        prepared_features = {}
        
        for feature_name in feature_names:
            feature_info = self._extract_feature_name_info(feature_name)
            
            if feature_info['type'] == 'original':
                if feature_info['base_feature'] in df.columns:
                    prepared_features[feature_name] = df[feature_info['base_feature']]
                else:
                    raise ValueError(f"필요한 피처가 없습니다: {feature_info['base_feature']}")
            
            elif feature_info['type'] == 'lag':
                prepared_features[feature_name] = self._create_lagged_features(
                    df, feature_info['base_feature'], 
                    feature_info['lag'], feature_info['time_minutes']
                )
            
            elif feature_info['type'] == 'roc':
                prepared_features[feature_name] = self._create_roc_features(
                    df, feature_info['base_feature'],
                    feature_info['roc_period'], feature_info['time_minutes'],
                    feature_info['agg_type']
                )
            
            else:
                logger.warning(f"알 수 없는 피처 타입: {feature_name}")
                prepared_features[feature_name] = 0  # 기본값
        
        # 데이터프레임 생성
        feature_df = pd.DataFrame(prepared_features, index=df.index)
        
        # 결측값 처리
        feature_df = feature_df.fillna(method='ffill').fillna(0)
        
        # 스케일링
        if scaler is not None:
            feature_df = pd.DataFrame(
                scaler.transform(feature_df),
                columns=feature_df.columns,
                index=feature_df.index
            )
        
        return feature_df
    
    def final_pred(self, short_model_pred: float, long_model_pred: float) -> float:
        """
        최종 예측값을 계산합니다.
        
        Args:
            short_model_pred: 단기 모델 예측값
            long_model_pred: 장기 모델 예측값
            
        Returns:
            최종 예측값
        """
        return short_model_pred + long_model_pred
    
    def _get_status(self, final_prediction: float) -> str:
        """
        예측값에 따른 상태를 반환합니다.
        
        Args:
            final_prediction: 최종 예측값
            
        Returns:
            상태 문자열
        """
        if final_prediction >= 68000:
            return "경고"
        else:
            return "정상"
    
    def predict(self, csv_path: str, target_time: Optional[Union[str, datetime]] = None) -> str:
        """
        CSV 파일과 목표 시간을 입력받아 추론을 수행하고 결과를 CSV에 추가합니다.
        
        Args:
            csv_path: 입력 CSV 파일 경로
            target_time: 목표 시간 (문자열 또는 datetime 객체, None이면 현재 시간)
            
        Returns:
            결과 메시지
        """
        try:
            # 목표 시간 처리
            if target_time is None:
                target_time = datetime.now()
            elif isinstance(target_time, str):
                target_time = pd.to_datetime(target_time)
            
            logger.info(f"목표 시간: {target_time}")
            
            # CSV 파일 읽기
            try:
                df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                logger.info(f"CSV 파일 로드 완료: {csv_path}")
            except Exception as e:
                return f"CSV 파일 읽기 오류: {str(e)}"
            
            # 입력 데이터 검증
            is_valid, error_msg = self._validate_input_csv(df)
            if not is_valid:
                return f"입력 데이터 검증 실패: {error_msg}"
            
            # 피처 검증
            is_valid, error_msg = self._validate_features(df)
            if not is_valid:
                return f"피처 검증 실패: {error_msg}"
            
            # 모델 검증
            if self.short_model is None or self.long_model is None:
                return "필요한 모델이 로드되지 않았습니다."
            
            # 피처 준비
            try:
                short_features = self._prepare_features(df, 'short')
                long_features = self._prepare_features(df, 'long')
                logger.info("피처 준비 완료")
            except Exception as e:
                return f"피처 준비 오류: {str(e)}"
            
            # 예측 수행
            try:
                short_pred = self.short_model.predict(short_features.iloc[-1:])[0]
                long_pred = self.long_model.predict(long_features.iloc[-1:])[0]
                logger.info(f"예측 완료 - 단기: {short_pred:.2f}, 장기: {long_pred:.2f}")
            except Exception as e:
                return f"예측 오류: {str(e)}"
            
            # 최종 예측값 계산
            final_prediction = self.final_pred(short_pred, long_pred)
            status = self._get_status(final_prediction)
            
            # 결과를 CSV에 추가
            try:
                result_df = pd.DataFrame({
                    '목표시간': [target_time],
                    '추론값': [final_prediction],
                    '비고': [status]
                })
                
                # 기존 결과 파일이 있으면 추가, 없으면 새로 생성
                result_file = csv_path.replace('.csv', '_results.csv')
                if os.path.exists(result_file):
                    existing_results = pd.read_csv(result_file)
                    updated_results = pd.concat([existing_results, result_df], ignore_index=True)
                else:
                    updated_results = result_df
                
                updated_results.to_csv(result_file, index=False)
                logger.info(f"결과 저장 완료: {result_file}")
                
                return f"추론 완료 - 최종값: {final_prediction:.2f}, 상태: {status}"
                
            except Exception as e:
                return f"결과 저장 오류: {str(e)}"
                
        except Exception as e:
            logger.error(f"예상치 못한 오류: {str(e)}")
            return f"시스템 오류: {str(e)}"

# 사용 예시 함수
def run_inference(csv_path: str, target_time: Optional[str] = None, model_dir: str = "./models"):
    """
    추론 시스템을 실행하는 편의 함수
    
    Args:
        csv_path: 입력 CSV 파일 경로
        target_time: 목표 시간 (문자열, None이면 현재 시간)
        model_dir: 모델 디렉토리 경로
        
    Returns:
        결과 메시지
    """
    try:
        system = InferenceSystem(model_dir)
        result = system.predict(csv_path, target_time)
        return result
    except Exception as e:
        return f"시스템 초기화 오류: {str(e)}"

if __name__ == "__main__":
    # 사용 예시
    csv_path = "input_data.csv"
    target_time = "2024-01-15 14:30:00"
    
    result = run_inference(csv_path, target_time)
    print(result)
