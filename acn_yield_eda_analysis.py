import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class ACNYieldAnalyzer:
    def __init__(self, data_path=None, df=None):
        """
        ACN 수율 최적화를 위한 EDA 분석 클래스
        
        Parameters:
        data_path: 데이터 파일 경로 (CSV, Excel 등)
        df: 이미 로드된 DataFrame
        """
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("데이터 경로 또는 DataFrame을 제공해야 합니다.")
        
        self.setup_data_types()
    
    def setup_data_types(self):
        """데이터 타입 설정 및 전처리"""
        # 날짜 컬럼 변환
        date_columns = ['batch_생산_날짜', '원료_투입_날짜']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # 범주형 변수 설정
        categorical_columns = ['원료_종류', '제품_버블링_유무', '냉동기_동시_가동_유무']
        for col in categorical_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
        
        # 수치형 변수 설정
        numeric_columns = [
            '폐기_촉매_재사용_횟수', '원료_품질값_a', '원료_품질값_b', '원료_품질값_c',
            '제품_품질값_a', '제품_품질값_b', '제품_품질값_c', '원료_투입량',
            '스팀_압력', '내온', '내압', '승온시간', '안정화_시간', '정제_시간',
            'CWS', 'CWS_압', 'CWR_온도', '현재_온도', 'product_level',
            '폐기_잔량', 'condenser_온도', '생산량', '수율'
        ]
        
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
    
    def basic_statistics(self):
        """기초 통계 분석"""
        print("=" * 80)
        print("ACN 수율 최적화 - 기초 통계 분석")
        print("=" * 80)
        
        # 1. 데이터 기본 정보
        print("\n1. 데이터 기본 정보")
        print("-" * 40)
        print(f"데이터 크기: {self.df.shape}")
        print(f"컬럼 수: {len(self.df.columns)}")
        print(f"행 수: {len(self.df)}")
        
        # 2. 결측치 분석
        print("\n2. 결측치 분석")
        print("-" * 40)
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            '결측치_수': missing_data,
            '결측치_비율(%)': missing_percent
        })
        missing_df = missing_df[missing_df['결측치_수'] > 0].sort_values('결측치_수', ascending=False)
        
        if len(missing_df) > 0:
            print(missing_df)
        else:
            print("결측치가 없습니다.")
        
        # 3. 수치형 변수 기초 통계
        print("\n3. 수치형 변수 기초 통계")
        print("-" * 40)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        basic_stats = self.df[numeric_cols].describe()
        print(basic_stats.round(3))
        
        # 4. 수율 관련 통계
        print("\n4. 수율 관련 통계")
        print("-" * 40)
        if '수율' in self.df.columns:
            yield_stats = self.df['수율'].describe()
            print("수율 통계:")
            print(yield_stats.round(3))
            
            print(f"\n수율 분포:")
            print(f"최고 수율: {self.df['수율'].max():.3f}")
            print(f"최저 수율: {self.df['수율'].min():.3f}")
            print(f"수율 범위: {self.df['수율'].max() - self.df['수율'].min():.3f}")
            print(f"수율 표준편차: {self.df['수율'].std():.3f}")
            print(f"수율 변동계수: {(self.df['수율'].std() / self.df['수율'].mean() * 100):.2f}%")
        
        # 5. 범주형 변수 분석
        print("\n5. 범주형 변수 분석")
        print("-" * 40)
        categorical_cols = self.df.select_dtypes(include=['category', 'object']).columns
        
        for col in categorical_cols:
            if col not in ['batch_생산_날짜', '원료_투입_날짜']:
                print(f"\n{col}:")
                value_counts = self.df[col].value_counts()
                print(value_counts)
                print(f"고유값 수: {self.df[col].nunique()}")
        
        return basic_stats
    
    def correlation_analysis(self):
        """상관관계 분석"""
        print("\n6. 수율과 주요 변수 간 상관관계 분석")
        print("-" * 50)
        
        if '수율' not in self.df.columns:
            print("수율 컬럼이 없습니다.")
            return None
        
        # 수치형 변수만 선택
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_data = self.df[numeric_cols].corr()
        
        # 수율과의 상관관계
        yield_corr = correlation_data['수율'].drop('수율').sort_values(key=abs, ascending=False)
        
        print("수율과의 상관계수 (절댓값 기준 정렬):")
        for var, corr in yield_corr.head(10).items():
            print(f"{var}: {corr:.4f}")
        
        return yield_corr
    
    def quality_level_analysis(self):
        """품질 측정 레벨 분석"""
        print("\n7. 품질 측정 레벨 분석")
        print("-" * 40)
        
        # 측정 레벨 관련 컬럼 찾기
        level_cols = [col for col in self.df.columns if 'level' in col.lower() or '레벨' in col]
        
        if level_cols:
            print("측정 레벨 컬럼들:")
            for col in level_cols:
                print(f"- {col}")
                if col in self.df.columns:
                    unique_levels = self.df[col].unique()
                    print(f"  고유 레벨: {sorted(unique_levels)}")
                    print(f"  레벨별 데이터 수:")
                    level_counts = self.df[col].value_counts().sort_index()
                    print(level_counts)
                    print()
        else:
            print("측정 레벨 관련 컬럼을 찾을 수 없습니다.")
    
    def batch_analysis(self):
        """배치별 분석"""
        print("\n8. 배치별 분석")
        print("-" * 40)
        
        if 'batch_생산_날짜' in self.df.columns:
            # 날짜별 배치 수
            daily_batches = self.df.groupby(self.df['batch_생산_날짜'].dt.date).size()
            print(f"일별 배치 수 통계:")
            print(daily_batches.describe())
            
            # 월별 배치 수
            monthly_batches = self.df.groupby(self.df['batch_생산_날짜'].dt.to_period('M')).size()
            print(f"\n월별 배치 수:")
            print(monthly_batches)
        
        # 원료 종류별 분석
        if '원료_종류' in self.df.columns and '수율' in self.df.columns:
            print(f"\n원료 종류별 수율 통계:")
            material_yield = self.df.groupby('원료_종류')['수율'].agg(['count', 'mean', 'std', 'min', 'max'])
            print(material_yield.round(3))
    
    def generate_summary_report(self):
        """종합 분석 리포트 생성"""
        print("\n" + "=" * 80)
        print("ACN 수율 최적화 - 종합 분석 리포트")
        print("=" * 80)
        
        # 기본 통계
        basic_stats = self.basic_statistics()
        
        # 상관관계 분석
        yield_corr = self.correlation_analysis()
        
        # 품질 레벨 분석
        self.quality_level_analysis()
        
        # 배치 분석
        self.batch_analysis()
        
        # 데이터 품질 요약
        print("\n9. 데이터 품질 요약")
        print("-" * 40)
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isnull().sum().sum()
        completeness = ((total_cells - missing_cells) / total_cells) * 100
        
        print(f"데이터 완성도: {completeness:.2f}%")
        print(f"전체 셀 수: {total_cells:,}")
        print(f"결측 셀 수: {missing_cells:,}")
        
        return {
            'basic_stats': basic_stats,
            'yield_correlation': yield_corr,
            'data_completeness': completeness
        }

# 사용 예시
def main():
    """
    메인 실행 함수
    실제 데이터 파일 경로를 입력하거나 DataFrame을 전달하세요.
    """
    
    # 예시: CSV 파일에서 데이터 로드
    # analyzer = ACNYieldAnalyzer(data_path='acn_batch_data.csv')
    
    # 예시: DataFrame 직접 전달
    # analyzer = ACNYieldAnalyzer(df=your_dataframe)
    
    # 분석 실행
    # results = analyzer.generate_summary_report()
    
    print("ACN 수율 최적화 분석을 시작하려면 데이터를 로드하세요.")
    print("사용법:")
    print("analyzer = ACNYieldAnalyzer(data_path='your_data.csv')")
    print("results = analyzer.generate_summary_report()")

if __name__ == "__main__":
    main()
