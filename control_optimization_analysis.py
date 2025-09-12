import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ControlOptimizationAnalyzer:
    """
    Control 변수 최적화 분석을 위한 클래스
    Control -> Target 관계의 특성과 예측 모델의 한계를 분석
    """
    
    def __init__(self, data=None):
        """
        데이터 초기화
        data: DataFrame with columns ['control', 'target'] or None for synthetic data
        """
        if data is not None:
            self.data = data.copy()
        else:
            self.data = self._generate_synthetic_data()
        
        self.control_col = 'control'
        self.target_col = 'target'
        
    def _generate_synthetic_data(self):
        """
        실제 상황을 모방한 합성 데이터 생성
        - Control 96, 97, 98: 충분한 샘플, 선형적 관계
        - Control 95, 94: 적은 샘플, 비선형적 관계
        """
        np.random.seed(42)
        
        data = []
        
        # Control 96, 97, 98: 충분한 샘플 (각각 100개)
        for control_val in [96, 97, 98]:
            n_samples = 100
            # 선형적 관계 + 노이즈
            target_vals = 2.5 * control_val + np.random.normal(0, 2, n_samples)
            for target in target_vals:
                data.append({'control': control_val, 'target': target})
        
        # Control 95, 94: 적은 샘플 (각각 15개)
        for control_val in [95, 94]:
            n_samples = 15
            # 비선형적 관계 + 더 큰 노이즈
            if control_val == 95:
                target_vals = 1.8 * control_val + 0.1 * (control_val - 95)**2 + np.random.normal(0, 5, n_samples)
            else:  # 94
                target_vals = 1.5 * control_val + 0.2 * (control_val - 94)**2 + np.random.normal(0, 6, n_samples)
            
            for target in target_vals:
                data.append({'control': control_val, 'target': target})
        
        return pd.DataFrame(data)
    
    def analyze_data_distribution(self):
        """데이터 분포 및 특성 분석"""
        print("=" * 60)
        print("1. 데이터 분포 및 특성 분석")
        print("=" * 60)
        
        # 기본 통계
        print(f"전체 데이터 크기: {len(self.data)}")
        print(f"Control 변수 범위: {self.data[self.control_col].min()} ~ {self.data[self.control_col].max()}")
        print(f"Target 변수 범위: {self.data[self.target_col].min():.2f} ~ {self.data[self.target_col].max():.2f}")
        
        # Control별 샘플 수
        print("\nControl별 샘플 수:")
        control_counts = self.data[self.control_col].value_counts().sort_index()
        for control_val, count in control_counts.items():
            print(f"  Control {control_val}: {count}개")
        
        # Control별 Target 통계
        print("\nControl별 Target 통계:")
        target_stats = self.data.groupby(self.control_col)[self.target_col].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        print(target_stats)
        
        return control_counts, target_stats
    
    def visualize_data_distribution(self):
        """데이터 분포 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Control별 Target 분포 (Box plot)
        axes[0, 0].boxplot([self.data[self.data[self.control_col] == val][self.target_col].values 
                           for val in sorted(self.data[self.control_col].unique())],
                          labels=sorted(self.data[self.control_col].unique()))
        axes[0, 0].set_title('Control별 Target 분포 (Box Plot)')
        axes[0, 0].set_xlabel('Control Value')
        axes[0, 0].set_ylabel('Target Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Control별 샘플 수
        control_counts = self.data[self.control_col].value_counts().sort_index()
        axes[0, 1].bar(control_counts.index, control_counts.values, 
                      color=['red' if x < 50 else 'blue' for x in control_counts.values])
        axes[0, 1].set_title('Control별 샘플 수')
        axes[0, 1].set_xlabel('Control Value')
        axes[0, 1].set_ylabel('Sample Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Scatter plot with regression line
        for control_val in sorted(self.data[self.control_col].unique()):
            subset = self.data[self.data[self.control_col] == control_val]
            color = 'red' if len(subset) < 50 else 'blue'
            axes[1, 0].scatter(subset[self.control_col], subset[self.target_col], 
                             alpha=0.6, label=f'Control {control_val}', color=color)
        
        # 전체 데이터에 대한 선형 회귀선
        X = self.data[[self.control_col]]
        y = self.data[self.target_col]
        lr = LinearRegression()
        lr.fit(X, y)
        X_pred = np.linspace(self.data[self.control_col].min(), 
                           self.data[self.control_col].max(), 100).reshape(-1, 1)
        y_pred = lr.predict(X_pred)
        axes[1, 0].plot(X_pred, y_pred, 'k--', linewidth=2, label='Linear Regression')
        
        axes[1, 0].set_title('Control vs Target (Scatter Plot)')
        axes[1, 0].set_xlabel('Control Value')
        axes[1, 0].set_ylabel('Target Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Control별 Target 분포 (Violin plot)
        data_for_violin = []
        labels_for_violin = []
        for control_val in sorted(self.data[self.control_col].unique()):
            subset = self.data[self.data[self.control_col] == control_val]
            data_for_violin.append(subset[self.target_col].values)
            labels_for_violin.append(f'Control {control_val}\n(n={len(subset)})')
        
        axes[1, 1].violinplot(data_for_violin, positions=range(len(data_for_violin)))
        axes[1, 1].set_xticks(range(len(labels_for_violin)))
        axes[1, 1].set_xticklabels(labels_for_violin, rotation=45)
        axes[1, 1].set_title('Control별 Target 분포 (Violin Plot)')
        axes[1, 1].set_ylabel('Target Value')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_sample_imbalance(self):
        """샘플 불균형 분석"""
        print("\n" + "=" * 60)
        print("2. 샘플 불균형 분석")
        print("=" * 60)
        
        control_counts = self.data[self.control_col].value_counts().sort_index()
        
        # 불균형 비율 계산
        max_count = control_counts.max()
        min_count = control_counts.min()
        imbalance_ratio = max_count / min_count
        
        print(f"최대 샘플 수: {max_count}")
        print(f"최소 샘플 수: {min_count}")
        print(f"불균형 비율: {imbalance_ratio:.2f}:1")
        
        # 각 Control 값의 전체 데이터에서 차지하는 비율
        print("\nControl별 데이터 비율:")
        for control_val, count in control_counts.items():
            ratio = count / len(self.data) * 100
            print(f"  Control {control_val}: {ratio:.1f}%")
        
        # 충분한 샘플 vs 부족한 샘플 구분
        sufficient_threshold = 50
        sufficient_samples = control_counts[control_counts >= sufficient_threshold]
        insufficient_samples = control_counts[control_counts < sufficient_threshold]
        
        print(f"\n충분한 샘플 (≥{sufficient_threshold}개): {list(sufficient_samples.index)}")
        print(f"부족한 샘플 (<{sufficient_threshold}개): {list(insufficient_samples.index)}")
        
        return imbalance_ratio, sufficient_samples, insufficient_samples
    
    def compare_linear_vs_nonlinear_models(self):
        """선형 vs 비선형 모델 성능 비교"""
        print("\n" + "=" * 60)
        print("3. 선형 vs 비선형 모델 성능 비교")
        print("=" * 60)
        
        X = self.data[[self.control_col]]
        y = self.data[self.target_col]
        
        # 모델 정의
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR (RBF)': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        results = {}
        
        for name, model in models.items():
            # 전체 데이터에 대한 성능
            model.fit(X, y)
            y_pred = model.predict(X)
            
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            cv_mse = -cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'MSE': mse,
                'R2': r2,
                'MAE': mae,
                'CV_MSE': cv_mse,
                'CV_std': cv_std,
                'predictions': y_pred
            }
            
            print(f"\n{name}:")
            print(f"  MSE: {mse:.2f}")
            print(f"  R²: {r2:.3f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  CV MSE: {cv_mse:.2f} (±{cv_std:.2f})")
        
        return results
    
    def analyze_prediction_errors_by_control(self, model_results):
        """Control별 예측 오차 분석"""
        print("\n" + "=" * 60)
        print("4. Control별 예측 오차 분석")
        print("=" * 60)
        
        X = self.data[[self.control_col]]
        y = self.data[self.target_col]
        
        error_analysis = {}
        
        for model_name, results in model_results.items():
            y_pred = results['predictions']
            errors = y - y_pred
            
            control_errors = {}
            for control_val in sorted(self.data[self.control_col].unique()):
                mask = self.data[self.control_col] == control_val
                control_error = errors[mask]
                
                control_errors[control_val] = {
                    'mean_error': control_error.mean(),
                    'std_error': control_error.std(),
                    'abs_mean_error': np.abs(control_error).mean(),
                    'sample_count': len(control_error)
                }
            
            error_analysis[model_name] = control_errors
        
        # 결과 출력
        for model_name, control_errors in error_analysis.items():
            print(f"\n{model_name} - Control별 예측 오차:")
            for control_val, errors in control_errors.items():
                print(f"  Control {control_val} (n={errors['sample_count']}): "
                      f"평균오차={errors['mean_error']:.2f}, "
                      f"절대평균오차={errors['abs_mean_error']:.2f}, "
                      f"오차표준편차={errors['std_error']:.2f}")
        
        return error_analysis
    
    def visualize_prediction_errors(self, model_results, error_analysis):
        """예측 오차 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        X = self.data[[self.control_col]]
        y = self.data[self.target_col]
        
        # 1. 모델별 예측 vs 실제값
        for i, (model_name, results) in enumerate(model_results.items()):
            y_pred = results['predictions']
            
            # Control별로 색상 구분
            for control_val in sorted(self.data[self.control_col].unique()):
                mask = self.data[self.control_col] == control_val
                subset_actual = y[mask]
                subset_pred = y_pred[mask]
                
                color = 'red' if len(subset_actual) < 50 else 'blue'
                axes[0, 0].scatter(subset_actual, subset_pred, 
                                 alpha=0.6, color=color, 
                                 label=f'Control {control_val}' if i == 0 else "")
            
            # 완벽한 예측선
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        axes[0, 0].set_title('예측값 vs 실제값 (모든 모델)')
        axes[0, 0].set_xlabel('실제값')
        axes[0, 0].set_ylabel('예측값')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Control별 절대평균오차
        control_vals = sorted(self.data[self.control_col].unique())
        for model_name, control_errors in error_analysis.items():
            abs_errors = [control_errors[val]['abs_mean_error'] for val in control_vals]
            axes[0, 1].plot(control_vals, abs_errors, 'o-', label=model_name, linewidth=2)
        
        axes[0, 1].set_title('Control별 절대평균오차')
        axes[0, 1].set_xlabel('Control Value')
        axes[0, 1].set_ylabel('절대평균오차')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Control별 오차 분포 (Box plot)
        for model_name, results in model_results.items():
            y_pred = results['predictions']
            errors = y - y_pred
            
            error_by_control = []
            labels = []
            for control_val in sorted(self.data[self.control_col].unique()):
                mask = self.data[self.control_col] == control_val
                control_error = errors[mask]
                error_by_control.append(control_error)
                labels.append(f'C{control_val}\n(n={len(control_error)})')
            
            # Box plot for each model
            bp = axes[1, 0].boxplot(error_by_control, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            break  # Only show one model for clarity
        
        axes[1, 0].set_title('Control별 예측 오차 분포')
        axes[1, 0].set_xlabel('Control Value')
        axes[1, 0].set_ylabel('예측 오차')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 샘플 수 vs 예측 성능 관계
        sample_counts = [len(self.data[self.data[self.control_col] == val]) 
                        for val in control_vals]
        
        for model_name, control_errors in error_analysis.items():
            abs_errors = [control_errors[val]['abs_mean_error'] for val in control_vals]
            axes[1, 1].scatter(sample_counts, abs_errors, label=model_name, s=100)
        
        axes[1, 1].set_title('샘플 수 vs 예측 성능')
        axes[1, 1].set_xlabel('샘플 수')
        axes[1, 1].set_ylabel('절대평균오차')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_comprehensive_report(self, control_counts, target_stats, imbalance_ratio, 
                                    model_results, error_analysis):
        """종합 분석 리포트 생성"""
        print("\n" + "=" * 80)
        print("종합 분석 리포트")
        print("=" * 80)
        
        print("\n📊 데이터 특성 요약:")
        print(f"  • 전체 데이터 크기: {len(self.data)}개")
        print(f"  • Control 변수 범위: {self.data[self.control_col].min()} ~ {self.data[self.control_col].max()}")
        print(f"  • 샘플 불균형 비율: {imbalance_ratio:.2f}:1")
        
        # 충분한 샘플 vs 부족한 샘플 구분
        sufficient_samples = control_counts[control_counts >= 50]
        insufficient_samples = control_counts[control_counts < 50]
        
        print(f"  • 충분한 샘플 (≥50개): {list(sufficient_samples.index)}")
        print(f"  • 부족한 샘플 (<50개): {list(insufficient_samples.index)}")
        
        print("\n🔍 주요 발견사항:")
        
        # 1. 샘플 불균형의 영향
        print("  1. 샘플 불균형 문제:")
        for control_val in insufficient_samples.index:
            count = control_counts[control_val]
            ratio = count / len(self.data) * 100
            print(f"     - Control {control_val}: {count}개 ({ratio:.1f}%) - 매우 적은 샘플")
        
        # 2. 모델 성능 차이
        print("\n  2. 모델 성능 차이:")
        linear_r2 = model_results['Linear Regression']['R2']
        rf_r2 = model_results['Random Forest']['R2']
        svr_r2 = model_results['SVR (RBF)']['R2']
        
        print(f"     - Linear Regression R²: {linear_r2:.3f}")
        print(f"     - Random Forest R²: {rf_r2:.3f}")
        print(f"     - SVR (RBF) R²: {svr_r2:.3f}")
        
        # 3. Control별 예측 오차 패턴
        print("\n  3. Control별 예측 오차 패턴:")
        for model_name, control_errors in error_analysis.items():
            print(f"     {model_name}:")
            for control_val in sorted(control_errors.keys()):
                abs_error = control_errors[control_val]['abs_mean_error']
                sample_count = control_errors[control_val]['sample_count']
                status = "높은 오차" if abs_error > 3.0 else "낮은 오차"
                print(f"       - Control {control_val} (n={sample_count}): {abs_error:.2f} ({status})")
        
        print("\n💡 결론 및 권장사항:")
        print("  1. 샘플 불균형이 예측 성능에 큰 영향을 미침")
        print("     - Control 96, 97, 98: 충분한 샘플로 안정적인 예측")
        print("     - Control 95, 94: 부족한 샘플로 불안정한 예측")
        
        print("\n  2. 비선형 모델의 한계:")
        print("     - 과적합 위험: 적은 샘플에 대해 복잡한 패턴 학습")
        print("     - 일반화 능력 부족: 새로운 데이터에 대한 예측 성능 저하")
        
        print("\n  3. 선형 모델의 한계:")
        print("     - 단순한 관계만 모델링 가능")
        print("     - 복잡한 비선형 관계 포착 불가")
        
        print("\n  4. 개선 방향:")
        print("     - 데이터 수집: Control 95, 94에 대한 더 많은 샘플 확보")
        print("     - 정규화 기법: L1/L2 정규화를 통한 과적합 방지")
        print("     - 앙상블 방법: 여러 모델의 예측 결과 결합")
        print("     - 도메인 지식 활용: Control-Target 관계에 대한 사전 지식 반영")

def main():
    """메인 실행 함수"""
    print("Control 최적화 분석 시작...")
    
    # 분석기 초기화
    analyzer = ControlOptimizationAnalyzer()
    
    # 1. 데이터 분포 분석
    control_counts, target_stats = analyzer.analyze_data_distribution()
    
    # 2. 데이터 시각화
    analyzer.visualize_data_distribution()
    
    # 3. 샘플 불균형 분석
    imbalance_ratio, sufficient_samples, insufficient_samples = analyzer.analyze_sample_imbalance()
    
    # 4. 모델 성능 비교
    model_results = analyzer.compare_linear_vs_nonlinear_models()
    
    # 5. 예측 오차 분석
    error_analysis = analyzer.analyze_prediction_errors_by_control(model_results)
    
    # 6. 예측 오차 시각화
    analyzer.visualize_prediction_errors(model_results, error_analysis)
    
    # 7. 종합 리포트 생성
    analyzer.generate_comprehensive_report(
        control_counts, target_stats, imbalance_ratio, 
        model_results, error_analysis
    )
    
    print("\n분석 완료!")

if __name__ == "__main__":
    main()
