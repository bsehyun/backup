import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class ACNVisualizationAnalyzer:
    def __init__(self, df):
        """
        ACN 수율 최적화를 위한 시각화 분석 클래스
        """
        self.df = df.copy()
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """플롯 스타일 설정"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def yield_distribution_analysis(self):
        """수율 분포 분석 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ACN 수율 분포 분석', fontsize=16, fontweight='bold')
        
        # 1. 히스토그램
        axes[0, 0].hist(self.df['수율'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(self.df['수율'].mean(), color='red', linestyle='--', 
                          label=f'평균: {self.df["수율"].mean():.3f}')
        axes[0, 0].axvline(self.df['수율'].median(), color='green', linestyle='--', 
                          label=f'중앙값: {self.df["수율"].median():.3f}')
        axes[0, 0].set_title('수율 히스토그램')
        axes[0, 0].set_xlabel('수율')
        axes[0, 0].set_ylabel('빈도')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 박스플롯
        axes[0, 1].boxplot(self.df['수율'], patch_artist=True, 
                          boxprops=dict(facecolor='lightblue', alpha=0.7))
        axes[0, 1].set_title('수율 박스플롯')
        axes[0, 1].set_ylabel('수율')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Q-Q 플롯 (정규성 검정)
        stats.probplot(self.df['수율'], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('수율 Q-Q 플롯 (정규성 검정)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 시간별 수율 추이
        if 'batch_생산_날짜' in self.df.columns:
            df_sorted = self.df.sort_values('batch_생산_날짜')
            axes[1, 1].plot(df_sorted['batch_생산_날짜'], df_sorted['수율'], 
                           marker='o', markersize=3, alpha=0.7)
            axes[1, 1].set_title('시간별 수율 추이')
            axes[1, 1].set_xlabel('생산 날짜')
            axes[1, 1].set_ylabel('수율')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def correlation_heatmap(self):
        """상관관계 히트맵"""
        # 수치형 변수만 선택
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # 상관계수 계산
        correlation_matrix = self.df[numeric_cols].corr()
        
        # 히트맵 생성
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={"shrink": .8})
        
        plt.title('ACN 공정 변수 상관관계 히트맵', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 수율과의 상관관계만 별도 표시
        if '수율' in correlation_matrix.columns:
            yield_corr = correlation_matrix['수율'].drop('수율').sort_values(key=abs, ascending=False)
            
            plt.figure(figsize=(12, 8))
            colors = ['red' if x < 0 else 'blue' for x in yield_corr.values]
            bars = plt.barh(range(len(yield_corr)), yield_corr.values, color=colors, alpha=0.7)
            plt.yticks(range(len(yield_corr)), yield_corr.index)
            plt.xlabel('상관계수')
            plt.title('수율과의 상관계수 (절댓값 기준 정렬)', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            
            # 상관계수 값 표시
            for i, (bar, value) in enumerate(zip(bars, yield_corr.values)):
                plt.text(value + (0.01 if value > 0 else -0.01), i, f'{value:.3f}', 
                        va='center', ha='left' if value > 0 else 'right')
            
            plt.tight_layout()
            plt.show()
    
    def process_parameter_analysis(self):
        """공정 파라미터 분석"""
        # 주요 공정 파라미터들
        process_params = ['내온', '내압', '스팀_압력', '승온시간', '안정화_시간', '정제_시간']
        available_params = [param for param in process_params if param in self.df.columns]
        
        if not available_params:
            print("공정 파라미터 컬럼을 찾을 수 없습니다.")
            return
        
        n_params = len(available_params)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, param in enumerate(available_params):
            row = i // n_cols
            col = i % n_cols
            
            # 산점도 (공정 파라미터 vs 수율)
            axes[row, col].scatter(self.df[param], self.df['수율'], alpha=0.6, s=30)
            axes[row, col].set_xlabel(param)
            axes[row, col].set_ylabel('수율')
            axes[row, col].set_title(f'{param} vs 수율')
            axes[row, col].grid(True, alpha=0.3)
            
            # 상관계수 표시
            corr = self.df[param].corr(self.df['수율'])
            axes[row, col].text(0.05, 0.95, f'상관계수: {corr:.3f}', 
                              transform=axes[row, col].transAxes, 
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 빈 subplot 제거
        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col])
        
        plt.suptitle('주요 공정 파라미터와 수율의 관계', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def material_quality_analysis(self):
        """원료 품질 분석"""
        if '원료_종류' not in self.df.columns:
            print("원료_종류 컬럼이 없습니다.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('원료 품질과 수율 분석', fontsize=16, fontweight='bold')
        
        # 1. 원료 종류별 수율 분포
        material_yield_data = []
        materials = self.df['원료_종류'].unique()
        for material in materials:
            material_data = self.df[self.df['원료_종류'] == material]['수율']
            material_yield_data.append(material_data)
        
        axes[0, 0].boxplot(material_yield_data, labels=materials, patch_artist=True)
        axes[0, 0].set_title('원료 종류별 수율 분포')
        axes[0, 0].set_ylabel('수율')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 원료 품질값과 수율의 관계
        quality_params = ['원료_품질값_a', '원료_품질값_b', '원료_품질값_c']
        available_quality = [param for param in quality_params if param in self.df.columns]
        
        if available_quality:
            for i, param in enumerate(available_quality[:2]):  # 최대 2개만 표시
                row = 0 if i == 0 else 1
                col = 1 if i == 0 else 0
                
                axes[row, col].scatter(self.df[param], self.df['수율'], alpha=0.6, s=30)
                axes[row, col].set_xlabel(param)
                axes[row, col].set_ylabel('수율')
                axes[row, col].set_title(f'{param} vs 수율')
                axes[row, col].grid(True, alpha=0.3)
                
                corr = self.df[param].corr(self.df['수율'])
                axes[row, col].text(0.05, 0.95, f'상관계수: {corr:.3f}', 
                                  transform=axes[row, col].transAxes,
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 3. 원료 투입량과 수율
        if '원료_투입량' in self.df.columns:
            axes[1, 1].scatter(self.df['원료_투입량'], self.df['수율'], alpha=0.6, s=30)
            axes[1, 1].set_xlabel('원료 투입량')
            axes[1, 1].set_ylabel('수율')
            axes[1, 1].set_title('원료 투입량 vs 수율')
            axes[1, 1].grid(True, alpha=0.3)
            
            corr = self.df['원료_투입량'].corr(self.df['수율'])
            axes[1, 1].text(0.05, 0.95, f'상관계수: {corr:.3f}', 
                          transform=axes[1, 1].transAxes,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def catalyst_reuse_analysis(self):
        """촉매 재사용 분석"""
        if '폐기_촉매_재사용_횟수' not in self.df.columns:
            print("폐기_촉매_재사용_횟수 컬럼이 없습니다.")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('촉매 재사용과 수율 분석', fontsize=16, fontweight='bold')
        
        # 1. 촉매 재사용 횟수 분포
        axes[0].hist(self.df['폐기_촉매_재사용_횟수'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0].set_title('촉매 재사용 횟수 분포')
        axes[0].set_xlabel('재사용 횟수')
        axes[0].set_ylabel('빈도')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 촉매 재사용 횟수 vs 수율
        axes[1].scatter(self.df['폐기_촉매_재사용_횟수'], self.df['수율'], alpha=0.6, s=30)
        axes[1].set_xlabel('촉매 재사용 횟수')
        axes[1].set_ylabel('수율')
        axes[1].set_title('촉매 재사용 횟수 vs 수율')
        axes[1].grid(True, alpha=0.3)
        
        # 상관계수 표시
        corr = self.df['폐기_촉매_재사용_횟수'].corr(self.df['수율'])
        axes[1].text(0.05, 0.95, f'상관계수: {corr:.3f}', 
                    transform=axes[1].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 3. 촉매 재사용 횟수별 평균 수율
        reuse_yield = self.df.groupby('폐기_촉매_재사용_횟수')['수율'].agg(['mean', 'std', 'count'])
        reuse_yield = reuse_yield[reuse_yield['count'] >= 3]  # 3개 이상 데이터가 있는 경우만
        
        axes[2].errorbar(reuse_yield.index, reuse_yield['mean'], 
                        yerr=reuse_yield['std'], marker='o', capsize=5, capthick=2)
        axes[2].set_xlabel('촉매 재사용 횟수')
        axes[2].set_ylabel('평균 수율')
        axes[2].set_title('촉매 재사용 횟수별 평균 수율')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def interactive_plotly_analysis(self):
        """인터랙티브 Plotly 분석"""
        # 수율과 주요 변수들의 3D 산점도
        if '내온' in self.df.columns and '내압' in self.df.columns:
            fig = px.scatter_3d(self.df, 
                               x='내온', y='내압', z='수율',
                               color='수율',
                               title='내온, 내압, 수율의 3D 관계',
                               labels={'내온': '내부 온도', '내압': '내부 압력', '수율': '수율'})
            fig.show()
        
        # 시간별 수율 추이 (인터랙티브)
        if 'batch_생산_날짜' in self.df.columns:
            df_sorted = self.df.sort_values('batch_생산_날짜')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_sorted['batch_생산_날짜'], 
                                   y=df_sorted['수율'],
                                   mode='lines+markers',
                                   name='수율',
                                   line=dict(color='blue', width=2),
                                   marker=dict(size=6)))
            
            # 이동평균 추가
            window_size = min(10, len(df_sorted) // 5)
            if window_size > 1:
                moving_avg = df_sorted['수율'].rolling(window=window_size).mean()
                fig.add_trace(go.Scatter(x=df_sorted['batch_생산_날짜'], 
                                       y=moving_avg,
                                       mode='lines',
                                       name=f'{window_size}점 이동평균',
                                       line=dict(color='red', width=2, dash='dash')))
            
            fig.update_layout(title='시간별 수율 추이 (인터랙티브)',
                             xaxis_title='생산 날짜',
                             yaxis_title='수율',
                             hovermode='x unified')
            fig.show()
    
    def generate_comprehensive_visualization(self):
        """종합 시각화 분석 실행"""
        print("ACN 수율 최적화 - 종합 시각화 분석을 시작합니다...")
        
        # 1. 수율 분포 분석
        print("1. 수율 분포 분석 중...")
        self.yield_distribution_analysis()
        
        # 2. 상관관계 분석
        print("2. 상관관계 분석 중...")
        self.correlation_heatmap()
        
        # 3. 공정 파라미터 분석
        print("3. 공정 파라미터 분석 중...")
        self.process_parameter_analysis()
        
        # 4. 원료 품질 분석
        print("4. 원료 품질 분석 중...")
        self.material_quality_analysis()
        
        # 5. 촉매 재사용 분석
        print("5. 촉매 재사용 분석 중...")
        self.catalyst_reuse_analysis()
        
        # 6. 인터랙티브 분석
        print("6. 인터랙티브 분석 중...")
        self.interactive_plotly_analysis()
        
        print("종합 시각화 분석이 완료되었습니다.")

# 사용 예시
def main():
    """
    메인 실행 함수
    """
    print("ACN 수율 최적화 시각화 분석을 시작하려면 데이터를 로드하세요.")
    print("사용법:")
    print("visualizer = ACNVisualizationAnalyzer(your_dataframe)")
    print("visualizer.generate_comprehensive_visualization()")

if __name__ == "__main__":
    main()
