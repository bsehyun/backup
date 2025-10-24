import os
import joblib
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Union, List

import numpy as np
import pandas as pd

from .tabular_service import TabularService
from .base_service import BaseService


np.random.seed(42)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WWTBlowerInferenceService(BaseService):
    """
    CSV 파일과 목표 시간을 입력받아 추론을 수행하고 결과를 CSV에 추가하는 시스템
    TabularService를 활용하여 사용자 업로드 데이터의 고유 식별 문제를 해결
    """

    def __init__(self, proj_path=None):
        """
        초기화
        
        Args:
            proj_path: 프로젝트 루트 경로 (선택사항)
        """
        self.model = None
        self.proj_path = proj_path or os.getenv("PROJECT_ROOT")
        self.model_path = os.path.join(self.proj_path, os.getenv("MODEL_DIR"), "wwt")
        self.save_path = os.path.join(self.proj_path, "app", os.getenv("DOWNLOAD_DIR"), "wwt")
        self.tags = []
        self.FE_tags = []
        self.control_FE_tag = ""
        self.control_tag = []
        self.resample_delta = "1d"
        
        # TabularService를 통한 데이터 관리
        self.tabular_service = TabularService

        # 모델과 로드
        self._load_models()

    def _load_models(self):
        """모델과 스케일러를 로드합니다."""
        try:
            filename = "model.joblib"
            filepath = os.path.join(self.model_path, filename)

            if os.path.exists(filepath):
                setattr(self, "model", joblib.load(filepath))
                logger.info(f"로드됨: {filename}")
            else:
                raise FileNotFoundError(f"파일을 찾을 수 없음: {filepath}")
                
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")


    def _create_origin_features(self, df: pd.DataFrame, feature_name: str, target_t: str) -> pd.Series:
        """
        피처를 생성합니다.
        
        Args:
            df: 입력 데이터프레임
            feature_name: 원천 피처 이름
            target_t: 추론 시간 
            
        Returns:
            생성된 피처
        """

        target_t = pd.to_datetime(target_t)
        window_start = target_t - pd.Timedelta(self.resample_delta)*2
        window_data = df.loc[(df.index > window_start)&(df.index <= target_t)]
        resample = window_data.resample(self.resample_delta, label="right", closed="right").mean()
        last_summary = resample.iloc[-1]

        return last_summary[feature_name]


    def _create_lagged_features(self, df: pd.DataFrame, feature_name: str, 
                               time_unit: int, target_t: str) -> pd.Series:
        """
        지연 피처를 생성합니다.
        
        Args:
            df: 입력 데이터프레임
            feature_name: 원천 피처 이름
            lag: 지연 값
            time_unit: 시간(단위)
            
        Returns:
            생성된 지연 피처
        """
        target_t = pd.to_datetime(target_t)
        window_start = target_t - pd.Timedelta(self.resample_delta)*2
        window_data = df.loc[(df.index > window_start)&(df.index <= target_t)]
        resample = window_data.resample(self.resample_delta, label="right", closed="right").mean()

        lag_vales = resample.iloc[-1-int(time_unit)]

        return lag_vales[feature_name]


    def _create_roc_features(self, df: pd.DataFrame, feature_name: str,
                           time_unit: int, target_t) -> pd.Series:
        """
        변화율 피처를 생성합니다.
        
        Args:
            df: 입력 데이터프레임
            feature_name: 원천 피처 이름
            roc_period: 변화율 계산 기간
            time_unit: 시간(단위)
            agg_type: 집계 타입 ('median')
            
        Returns:
            생성된 변화율 피처
        """
        target_t = pd.to_datetime(target_t)
        window_start = target_t - pd.Timedelta(self.resample_delta)*2
        window_data = df.loc[(df.index > window_start)&(df.index <= target_t)]
        resample = window_data.resample(self.resample_delta, label="right", closed="right").mean()

        roll_vales = resample.iloc[-int(time_unit):].mean()

        return roll_vales[feature_name]


    def _extract_feature_name_info(self, feature_name: str) -> Dict:
        """
        피처 이름에서 변형 정보를 추출합니다.
        
        Args:
            feature_name: 피처 이름
            
        Returns:
            변형 정보 딕셔너리
        """
        # 원천 피처만 있는 경우
        if feature_name in self.tags:
            return {
                'base_feature': feature_name,
                'type': 'original'
            }

        # 지연 피처인 경우: {원천feature이름}_lag{시간}
        if "_lag" in feature_name :
            try:
                time_unit = int(feature_name.split("_lag")[-1]) 
                base_feature = feature_name.split("_lag")[0]
                return {
                    'base_feature': base_feature,
                    'type': 'lag',
                    'time_unit': time_unit
                }
            except ValueError:
                pass

        # 변화율 피처인 경우: {원천feature이름}_roll{시간}_{mean}
        if "_roll" in feature_name:
            try:
                time_unit = int(feature_name.split("_roll")[-1].split("_")[0])
                agg_type = feature_name.split("_")[-1]
                base_feature = feature_name.split("_roll")[0]
                
                if agg_type in ['mean']:
                    return {
                        'base_feature': base_feature,
                        'type': 'roc',
                        'time_unit': time_unit,
                        'agg_type': agg_type
                    }
            except ValueError:
                pass

        # 파싱할 수 없는 경우
        return {
            'base_feature': feature_name,
            'type': 'unknown'
        }


    def _prepare_features(self, df: pd.DataFrame, target_t: str) -> pd.DataFrame:
        """
        모델에 필요한 피처를 준비합니다.
        
        Args:
            df: 입력 데이터프레임
            
        Returns:
            준비된 피처 데이터프레임
        """

        # 피처 준비
        prepared_features = {}

        if not set(self.tags + self.control_tag).issubset(df.columns):
            raise ValueError(f"Not ehough tags")

        for feature_name in self.FE_tags:
            if feature_name == self.control_FE_tag:
                df["UC_목표값백분율"] = (df[self.control_tag].sum(axis=1))/4

            feature_info = self._extract_feature_name_info(feature_name)

            if feature_info['type'] == 'original':

                if feature_info['base_feature'] in df.columns:
                    prepared_features[feature_name] = self._create_origin_features(
                        df, feature_info['base_feature'],
                        target_t
                    )
                else:
                    raise ValueError(f"필요한 피처가 없습니다: {feature_info['base_feature']}")

            elif feature_info['type'] == 'lag':
                prepared_features[feature_name] = self._create_lagged_features(
                    df, feature_info['base_feature'], 
                    feature_info['time_unit'], target_t
                )

            elif feature_info['type'] == 'roc':
                prepared_features[feature_name] = self._create_roc_features(
                    df, feature_info['base_feature'],
                    feature_info['time_unit'], target_t
                )

            else:
                raise ValueError(f"알 수 없는 피처 타입: {feature_name}")  # 기본값
        # 데이터프레임 생성
        feature_df = pd.DataFrame(
            columns = [target_t],
            index = prepared_features.keys(),
            data = prepared_features.values()
        ).T

        return feature_df


    def _preprocess(self, data_ids: List[str] = None, target_time: Optional[Union[str, datetime]] = None) -> tuple:
        """
        TabularService를 통해 업로드된 데이터를 읽어 추론 dataframe을 생성합니다.
        
        Args:
            data_ids: 사용할 데이터 ID 목록 (None이면 최신 WWT 데이터 사용)
            target_time: 목표 시간 (문자열 또는 datetime 객체, None이면 현재 시간)
            
        Returns:
            tuple: (features DataFrame, target_time)
        """
        try:
            # 데이터 ID가 제공되지 않으면 최신 WWT 데이터 사용
            if data_ids is None:
                wwt_data_list = self.tabular_service.list_by_category("WWT")
                if not wwt_data_list:
                    raise FileNotFoundError("WWT 카테고리 데이터가 없습니다.")
                # 최신 데이터부터 정렬하여 사용
                wwt_data_list.sort(key=lambda x: x.uploaded_at, reverse=True)
                data_ids = [data.id for data in wwt_data_list[:4]]  # 최대 4개 파일 사용

            # CSV 파일 읽기
            try:
                dfs = []
                for data_id in data_ids:
                    tabular_data = self.tabular_service.get_by_id(data_id)
                    if tabular_data is None:
                        logger.warning(f"데이터를 찾을 수 없습니다: {data_id}")
                        continue
                    
                    df = self.tabular_service.read_csv(tabular_data, index_col=0, encoding="ANSI")
                    dfs.append(df)
                    
                if not dfs:
                    raise FileNotFoundError("유효한 데이터가 없습니다.")
                    
            except Exception as e:
                raise FileNotFoundError(f"CSV 파일 읽기 오류: {str(e)}")

            df = pd.concat(dfs)
            df.index = pd.to_datetime(df.index)
            df = df.resample("1h", closed="right", label="right").mean()

            # 목표 시간 처리
            if target_time is None:
                target_time = pd.to_datetime(df.index[-1]).replace(hour=10)
            elif isinstance(target_time, str):
                target_time = pd.to_datetime(target_time).floor("10min")

            logger.info(f"목표 시간: {target_time}")

            # 피처 준비
            features = self._prepare_features(df, target_time)
            logger.info("피처 준비 완료")

            return features, target_time
        except Exception as e:
            logger.error(f"예상치 못한 오류: {str(e)}")
            raise RuntimeError(f"시스템 오류: {str(e)}")


    def run(self, data_ids: List[str] = None, target_time: Optional[Union[str, datetime]] = None) -> Dict[str, any]:    
        """
        TabularService를 통해 데이터를 읽어 추론을 수행하고 결과를 반환합니다.
        
        Args:
            data_ids: 사용할 데이터 ID 목록 (None이면 최신 WWT 데이터 사용)
            target_time: 목표 시간 (문자열 또는 datetime 객체, None이면 현재 시간)
            
        Returns:
            Dict: 추론 결과 및 메타데이터
        """
        try:
            # 데이터 전처리
            try:
                df, target_time = self._preprocess(data_ids, target_time)
            except Exception as e:
                raise FileNotFoundError(f"데이터 읽기 오류: {str(e)}")

            logger.info(f"목표 시간: {target_time}")

            if target_time not in df.index:
                raise ValueError("타겟 타임이 데이터에 존재하지 않습니다.")

            if (df.isnull().sum().sum() > 0):
                raise ValueError("추론 행에 결측치가 있습니다.")

            # 추론 수행
            pred_actual = self.model.predict(df)[0] / 10

            target_df_if96 = df.copy()
            target_df_if96[self.control_FE_tag] = 96.0
            pred_if96 = self.model.predict(target_df_if96)[0] / 10

            target_df_if97 = df.copy()
            target_df_if97[self.control_FE_tag] = 97.0
            pred_if97 = self.model.predict(target_df_if97)[0] / 10

            target_df_if98 = df.copy()
            target_df_if98[self.control_FE_tag] = 98.0
            pred_if98 = self.model.predict(target_df_if98)[0] / 10

            logger.info(f"예측 완료 \n실제 DO : {pred_actual:.2f} \n 96% DO: {pred_if96:.2f}\n 97% DO: {pred_if97:.2f}\n 98% DO: {pred_if98:.2f}")

            # 결과 데이터프레임 생성
            result_df = pd.DataFrame({
                'Time': [target_time],
                'DO': [f"{pred_actual:.2f}"],
                '96_DO': [f"{pred_if96:.2f}"],
                '97_DO': [f"{pred_if97:.2f}"],
                '98_DO': [f"{pred_if98:.2f}"],
            })

            # 결과를 TabularService를 통해 저장
            result_filename = f"wwt_result_{self.uid()}.csv"
            result_file_path = os.path.join(self.save_path, result_filename)
            os.makedirs(self.save_path, exist_ok=True)
            
            result_df.to_csv(result_file_path, index=False, encoding="utf-8-sig")
            logger.info(f"결과 저장 완료: {result_file_path}")

            return {
                "success": True,
                "target_time": target_time.isoformat() if isinstance(target_time, pd.Timestamp) else str(target_time),
                "predictions": {
                    "actual_do": float(pred_actual),
                    "do_96": float(pred_if96),
                    "do_97": float(pred_if97),
                    "do_98": float(pred_if98)
                },
                "result_file": result_file_path,
                "data_ids_used": data_ids
            }

        except Exception as e:
            logger.error(f"예상치 못한 오류: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
