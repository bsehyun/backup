import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesAnalyzer:
    def __init__(self):
        self.data = None
        self.analysis_results = {}
        
    def load_data(self, data, column_name='C'):
        """
        데이터 로드
        data: pandas DataFrame 또는 numpy array
        column_name: C 변수의 컬럼명
        """
        if isinstance(data, np.ndarray):
            self.data = pd.DataFrame(data, columns=[column_name])
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise ValueError("데이터는 numpy array 또는 pandas DataFrame이어야 합니다.")
        
        print(f"데이터 로드 완료: {len(self.data)} 개의 관측치")
        return self
    
    def basic_statistics(self):
        """기본 통계 분석"""
        print("=== 기본 통계 분석 ===")
        print(self.data.describe())
        
        # 분포 시각화
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 히스토그램
        axes[0, 0].hist(self.data.iloc[:, 0], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('분포 히스토그램')
        axes[0, 0].set_xlabel('값')
        axes[0, 0].set_ylabel('빈도')
        
        # 시계열 플롯
        axes[0, 1].plot(self.data.iloc[:, 0])
        axes[0, 1].set_title('시계열 플롯')
        axes[0, 1].set_xlabel('시간')
        axes[0, 1].set_ylabel('값')
        
        # 박스플롯
        axes[1, 0].boxplot(self.data.iloc[:, 0])
        axes[1, 0].set_title('박스플롯')
        
        # Q-Q 플롯
        stats.probplot(self.data.iloc[:, 0], dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q 플롯 (정규성 검정)')
        
        plt.tight_layout()
        plt.show()
        
        return self
    
    def stationarity_test(self):
        """정상성 검정"""
        print("=== 정상성 검정 ===")
        
        series = self.data.iloc[:, 0]
        
        # ADF 검정
        adf_result = adfuller(series)
        print(f"ADF 검정:")
        print(f"  통계량: {adf_result[0]:.6f}")
        print(f"  p-value: {adf_result[1]:.6f}")
        print(f"  정상성: {'정상' if adf_result[1] < 0.05 else '비정상'}")
        
        # KPSS 검정
        kpss_result = kpss(series)
        print(f"\nKPSS 검정:")
        print(f"  통계량: {kpss_result[0]:.6f}")
        print(f"  p-value: {kpss_result[1]:.6f}")
        print(f"  정상성: {'정상' if kpss_result[1] > 0.05 else '비정상'}")
        
        self.analysis_results['stationarity'] = {
            'adf': adf_result,
            'kpss': kpss_result
        }
        
        return self
    
    def autocorrelation_analysis(self, max_lag=50):
        """자기상관 분석"""
        print("=== 자기상관 분석 ===")
        
        series = self.data.iloc[:, 0]
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # ACF (자기상관함수)
        plot_acf(series, lags=max_lag, ax=axes[0])
        axes[0].set_title('ACF (자기상관함수)')
        
        # PACF (부분자기상관함수)
        plot_pacf(series, lags=max_lag, ax=axes[1])
        axes[1].set_title('PACF (부분자기상관함수)')
        
        plt.tight_layout()
        plt.show()
        
        # 자기상관 계수 계산
        autocorr = pd.Series(series).autocorr(lag=1)
        print(f"1차 자기상관 계수: {autocorr:.6f}")
        
        return self
    
    def seasonal_decomposition(self, period=None):
        """계절성 분해"""
        print("=== 계절성 분해 ===")
        
        series = self.data.iloc[:, 0]
        
        # period가 지정되지 않은 경우 자동 추정
        if period is None:
            # 데이터 길이의 1/4을 기본값으로 사용
            period = max(2, len(series) // 4)
        
        try:
            decomposition = seasonal_decompose(series, period=period, extrapolate_trend='freq')
            
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            
            decomposition.observed.plot(ax=axes[0])
            axes[0].set_title('원본 데이터')
            
            decomposition.trend.plot(ax=axes[1])
            axes[1].set_title('추세')
            
            decomposition.seasonal.plot(ax=axes[2])
            axes[2].set_title('계절성')
            
            decomposition.resid.plot(ax=axes[3])
            axes[3].set_title('잔차')
            
            plt.tight_layout()
            plt.show()
            
            # 계절성 강도 계산
            seasonal_strength = np.std(decomposition.seasonal) / np.std(decomposition.resid)
            print(f"계절성 강도: {seasonal_strength:.6f}")
            
        except Exception as e:
            print(f"계절성 분해 실패: {e}")
        
        return self
    
    def pattern_detection(self):
        """패턴 탐지"""
        print("=== 패턴 탐지 ===")
        
        series = self.data.iloc[:, 0]
        
        # 1. 주기성 탐지 (FFT)
        fft = np.fft.fft(series)
        freqs = np.fft.fftfreq(len(series))
        
        # 주요 주파수 찾기
        power = np.abs(fft) ** 2
        main_freq_idx = np.argmax(power[1:len(power)//2]) + 1
        main_period = 1 / freqs[main_freq_idx] if freqs[main_freq_idx] != 0 else float('inf')
        
        print(f"주요 주기: {main_period:.2f} 단위")
        
        # 2. 변동점 탐지
        # 이동평균과의 차이로 변동점 탐지
        window = min(20, len(series) // 10)
        rolling_mean = series.rolling(window=window).mean()
        change_points = np.where(np.abs(series - rolling_mean) > 2 * series.std())[0]
        
        print(f"변동점 개수: {len(change_points)}")
        
        # 3. 시각화
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # FFT 결과
        axes[0].plot(freqs[1:len(freqs)//2], power[1:len(power)//2])
        axes[0].set_title('주파수 스펙트럼')
        axes[0].set_xlabel('주파수')
        axes[0].set_ylabel('파워')
        
        # 변동점 표시
        axes[1].plot(series, label='원본 데이터')
        axes[1].plot(rolling_mean, label='이동평균', alpha=0.7)
        if len(change_points) > 0:
            axes[1].scatter(change_points, series.iloc[change_points], 
                           color='red', label='변동점', zorder=5)
        axes[1].set_title('변동점 탐지')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return self
    
    def forecasting_ability_test(self, test_size=0.2):
        """예측 가능성 테스트"""
        print("=== 예측 가능성 테스트 ===")
        
        series = self.data.iloc[:, 0]
        train_size = int(len(series) * (1 - test_size))
        
        train = series[:train_size]
        test = series[train_size:]
        
        # 간단한 ARIMA 모델로 예측 성능 테스트
        try:
            # ARIMA(1,1,1) 모델
            model = ARIMA(train, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # 예측
            forecast = fitted_model.forecast(steps=len(test))
            
            # 성능 평가
            mse = np.mean((test - forecast) ** 2)
            mae = np.mean(np.abs(test - forecast))
            
            print(f"예측 성능:")
            print(f"  MSE: {mse:.6f}")
            print(f"  MAE: {mae:.6f}")
            print(f"  상대 오차: {mae/np.mean(test)*100:.2f}%")
            
            # 시각화
            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train, label='훈련 데이터')
            plt.plot(test.index, test, label='실제 테스트 데이터')
            plt.plot(test.index, forecast, label='예측', linestyle='--')
            plt.title('예측 성능 테스트')
            plt.legend()
            plt.show()
            
        except Exception as e:
            print(f"예측 모델링 실패: {e}")
        
        return self
    
    def comprehensive_report(self):
        """종합 분석 리포트"""
        print("=== 종합 분석 리포트 ===")
        
        series = self.data.iloc[:, 0]
        
        report = {
            "데이터 길이": len(series),
            "평균": series.mean(),
            "표준편차": series.std(),
            "변동계수": series.std() / series.mean(),
            "최대값": series.max(),
            "최소값": series.min(),
            "범위": series.max() - series.min(),
            "1차 자기상관": series.autocorr(lag=1),
            "정상성": "정상" if self.analysis_results.get('stationarity', {}).get('adf', [0, 1])[1] < 0.05 else "비정상"
        }
        
        print("\n=== 요약 통계 ===")
        for key, value in report.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
        
        # 패턴 존재 여부 판단
        autocorr = abs(report["1차 자기상관"])
        cv = report["변동계수"]
        
        print("\n=== 패턴 분석 결과 ===")
        if autocorr > 0.3:
            print("✓ 자기상관 패턴이 강함 (예측 가능성 높음)")
        elif autocorr > 0.1:
            print("○ 자기상관 패턴이 약함 (예측 가능성 보통)")
        else:
            print("✗ 자기상관 패턴이 없음 (예측 어려움)")
        
        if cv < 0.1:
            print("✓ 변동성이 낮음 (안정적)")
        elif cv < 0.5:
            print("○ 변동성이 보통")
        else:
            print("✗ 변동성이 높음 (불안정)")
        
        return report

# 사용 예시
def example_usage():
    """사용 예시"""
    # 샘플 데이터 생성 (실제 데이터로 교체)
    np.random.seed(42)
    n = 1000
    
    # 1. 랜덤 데이터 (패턴 없음)
    random_data = np.random.normal(0, 1, n)
    
    # 2. 트렌드가 있는 데이터
    trend_data = np.linspace(0, 10, n) + np.random.normal(0, 0.5, n)
    
    # 3. 계절성이 있는 데이터
    seasonal_data = 5 * np.sin(2 * np.pi * np.arange(n) / 50) + np.random.normal(0, 0.5, n)
    
    # 분석 실행
    analyzer = TimeSeriesAnalyzer()
    
    print("=== 랜덤 데이터 분석 ===")
    analyzer.load_data(random_data, 'C_random')
    analyzer.basic_statistics()
    analyzer.stationarity_test()
    analyzer.autocorrelation_analysis()
    analyzer.pattern_detection()
    analyzer.comprehensive_report()
    
    print("\n=== 트렌드 데이터 분석 ===")
    analyzer.load_data(trend_data, 'C_trend')
    analyzer.basic_statistics()
    analyzer.stationarity_test()
    analyzer.autocorrelation_analysis()
    analyzer.pattern_detection()
    analyzer.comprehensive_report()
    
    print("\n=== 계절성 데이터 분석 ===")
    analyzer.load_data(seasonal_data, 'C_seasonal')
    analyzer.basic_statistics()
    analyzer.stationarity_test()
    analyzer.autocorrelation_analysis()
    analyzer.seasonal_decomposition()
    analyzer.pattern_detection()
    analyzer.forecasting_ability_test()
    analyzer.comprehensive_report()

if __name__ == "__main__":
    example_usage() 
