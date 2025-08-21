# 시계열 데이터 EDA (Exploratory Data Analysis) 도구

대용량 고차원 시계열 데이터의 첫 번째 데이터 성질을 체계적으로 분석하는 도구입니다.

## 📋 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [설치 및 설정](#설치-및-설정)
- [사용법](#사용법)
- [파일 구조](#파일-구조)
- [예시](#예시)
- [API 문서](#api-문서)

## 🎯 개요

이 도구는 다음과 같은 상황에서 유용합니다:

- **대용량 시계열 데이터**: 수천 개의 시점과 수백 개의 변수를 가진 데이터
- **고차원 데이터**: 변수 수가 많은 다차원 시계열 데이터
- **첫 번째 데이터 성질 파악**: 시계열의 초기 상태와 특성을 이해하고자 할 때
- **체계적인 EDA**: 반복 가능하고 일관된 탐색적 데이터 분석

## ✨ 주요 기능

### 1. 첫 번째 시점 분석
- 첫 번째 시점의 통계적 특성 분석
- 분포 시각화 (히스토그램, 박스플롯, Q-Q 플롯)
- 이상치 탐지 (IQR, Z-score 방법)
- 값 범위별 분류 및 분석

### 2. 차원 분석
- PCA를 통한 차원 축소 분석
- 설명 분산 비율 계산
- 첫 번째 시점의 주성분 분석
- 차원 축소 효율성 평가

### 3. 변수 특성 분석
- 변수별 값 크기 분류
- 상위/하위 값 변수 식별
- 0값, 음수, 양수 변수 분류
- 변수 그룹화 및 클러스터링

### 4. 시간적 패턴 분석
- 초기 구간의 시계열 특성 분석
- 트렌드, 계절성, 변동성 패턴 식별
- 자기상관 분석
- 변동성 높은 변수 탐지

### 5. 상관관계 분석
- 변수 간 상관관계 계산
- 첫 번째 시점과의 상관관계 분석
- 상관관계 히트맵 시각화
- 높은 상관관계 변수 그룹 식별

### 6. 종합 보고서 생성
- 분석 결과 요약
- 시각화 결과 저장
- JSON 및 텍스트 형태의 결과 저장
- 재현 가능한 분석 워크플로우

## 🚀 설치 및 설정

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 필요한 패키지들

```
pandas>=1.5.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
scipy>=1.10.0
umap-learn>=0.5.3
numpy>=1.24.3
```

## 📖 사용법

### 기본 사용법

```python
from timeseries_eda import TimeseriesEDA

# 1. 데이터 로드
eda = TimeseriesEDA(data_path='your_data.csv')
# 또는
eda = TimeseriesEDA(df=your_dataframe)

# 2. 전체 분석 실행
eda.run_full_analysis(output_dir='results', save_plots=True)
```

### 단계별 분석

```python
# 1. 첫 번째 시점 분석
first_stats = eda.analyze_first_timestep(save_plots=True)

# 2. 차원 분석
eda.analyze_dimensions(save_plots=True)

# 3. 변수 특성 분석
eda.analyze_variable_characteristics()

# 4. 초기 시계열 분석
eda.analyze_initial_timeseries(n_periods=100)

# 5. 상관관계 분석
eda.analyze_correlations(save_plots=True)

# 6. 변수 클러스터링
eda.cluster_variables(n_clusters=5)
```

### 유틸리티 함수 사용

```python
from timeseries_eda_utils import *

# 샘플 데이터 생성
sample_df = generate_sample_timeseries_data(n_timesteps=1000, n_features=200)

# 빠른 개요 확인
overview = quick_data_overview(sample_df)

# 첫 번째 시점 상세 분석
first_stats = analyze_first_timestep_detailed(sample_df)

# 변수 그룹 분석
groups = analyze_variable_groups(sample_df, n_groups=5)

# 이상치 탐지
outliers = detect_outliers_in_first_timestep(sample_df, method='iqr')

# 시간적 패턴 분석
temporal_patterns = analyze_temporal_patterns(sample_df)

# 종합 시각화
create_summary_visualization(sample_df)
```

## 📁 파일 구조

```
timeseries_eda/
├── timeseries_eda.py          # 메인 EDA 클래스
├── timeseries_eda_utils.py    # 유틸리티 함수들
├── example_usage.py           # 사용 예시
├── requirements.txt           # 필요한 패키지 목록
└── README_timeseries_eda.md   # 이 파일
```

## 🎨 예시

### 예시 1: 기본 사용법

```python
# 샘플 데이터 생성 및 분석
from timeseries_eda import TimeseriesEDA
from timeseries_eda_utils import generate_sample_timeseries_data

# 데이터 생성
sample_df = generate_sample_timeseries_data(n_timesteps=1000, n_features=200)

# EDA 분석
eda = TimeseriesEDA(df=sample_df)
eda.run_full_analysis(output_dir='my_results')
```

### 예시 2: 실제 데이터 분석

```python
import pandas as pd
from timeseries_eda import TimeseriesEDA

# 실제 데이터 로드
df = pd.read_csv('your_timeseries_data.csv')

# EDA 분석
eda = TimeseriesEDA(df=df)

# 첫 번째 시점 상세 분석
first_stats = eda.analyze_first_timestep_detailed(save_plots=True)

# 차원 분석 (고차원 데이터인 경우)
if df.shape[1] > 100:
    eda.analyze_dimensions(save_plots=True)

# 결과 확인
print(f"첫 번째 시점 평균: {first_stats['mean']:.4f}")
print(f"첫 번째 시점 표준편차: {first_stats['std']:.4f}")
```

### 예시 3: 커스텀 분석

```python
from timeseries_eda_utils import *

# 커스텀 분석 함수
def custom_analysis(df):
    first_row = df.iloc[0]
    
    # 특정 조건에 맞는 변수들 찾기
    high_value_vars = first_row[first_row > first_row.quantile(0.9)]
    low_value_vars = first_row[first_row < first_row.quantile(0.1)]
    
    print(f"높은 값 변수: {len(high_value_vars)}개")
    print(f"낮은 값 변수: {len(low_value_vars)}개")
    
    return high_value_vars, low_value_vars

# 분석 실행
sample_df = generate_sample_timeseries_data()
high_vars, low_vars = custom_analysis(sample_df)
```

## 📚 API 문서

### TimeseriesEDA 클래스

#### 초기화
```python
TimeseriesEDA(data_path=None, df=None)
```

**매개변수:**
- `data_path` (str): 데이터 파일 경로 (.csv, .parquet, .xlsx)
- `df` (pd.DataFrame): 직접 전달된 데이터프레임

#### 주요 메서드

##### analyze_first_timestep()
첫 번째 시점 데이터 분석

**매개변수:**
- `save_plots` (bool): 플롯 저장 여부 (기본값: True)
- `output_dir` (str): 결과 저장 디렉토리 (기본값: 'eda_results')

**반환값:**
- `dict`: 첫 번째 시점 통계 정보

##### analyze_dimensions()
차원 분석 (PCA 등)

**매개변수:**
- `save_plots` (bool): 플롯 저장 여부 (기본값: True)
- `output_dir` (str): 결과 저장 디렉토리 (기본값: 'eda_results')

##### analyze_variable_characteristics()
변수별 특성 분석

**반환값:**
- `dict`: 변수 특성 분석 결과

##### analyze_initial_timeseries()
초기 시계열 구간 분석

**매개변수:**
- `n_periods` (int): 분석할 초기 구간 길이 (기본값: 100)
- `save_plots` (bool): 플롯 저장 여부 (기본값: True)
- `output_dir` (str): 결과 저장 디렉토리 (기본값: 'eda_results')

##### analyze_correlations()
상관관계 분석

**매개변수:**
- `save_plots` (bool): 플롯 저장 여부 (기본값: True)
- `output_dir` (str): 결과 저장 디렉토리 (기본값: 'eda_results')

##### cluster_variables()
변수 클러스터링

**매개변수:**
- `n_clusters` (int): 클러스터 수 (기본값: 5)
- `save_plots` (bool): 플롯 저장 여부 (기본값: True)
- `output_dir` (str): 결과 저장 디렉토리 (기본값: 'eda_results')

##### run_full_analysis()
전체 분석 실행

**매개변수:**
- `output_dir` (str): 결과 저장 디렉토리 (기본값: 'eda_results')
- `save_plots` (bool): 플롯 저장 여부 (기본값: True)

### 유틸리티 함수들

#### generate_sample_timeseries_data()
샘플 시계열 데이터 생성

**매개변수:**
- `n_timesteps` (int): 시계열 길이 (기본값: 1000)
- `n_features` (int): 변수 수 (기본값: 200)
- `noise_level` (float): 노이즈 수준 (기본값: 0.1)

**반환값:**
- `pd.DataFrame`: 샘플 시계열 데이터

#### quick_data_overview()
데이터 빠른 개요 확인

**매개변수:**
- `df` (pd.DataFrame): 분석할 데이터프레임

**반환값:**
- `dict`: 데이터 개요 정보

#### analyze_first_timestep_detailed()
첫 번째 시점 상세 분석

**매개변수:**
- `df` (pd.DataFrame): 분석할 데이터프레임
- `save_plots` (bool): 플롯 저장 여부 (기본값: False)
- `output_dir` (str): 저장 디렉토리 (기본값: '.')

**반환값:**
- `dict`: 상세 분석 결과

## 🔧 고급 사용법

### 1. 메모리 효율적인 분석

대용량 데이터의 경우 메모리 사용량을 고려해야 합니다:

```python
# 청크 단위로 분석
chunk_size = 10000
for chunk in pd.read_csv('large_data.csv', chunksize=chunk_size):
    # 청크별 분석
    eda = TimeseriesEDA(df=chunk)
    eda.analyze_first_timestep()
```

### 2. 병렬 처리

여러 데이터셋을 동시에 분석:

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def analyze_dataset(data_path):
    eda = TimeseriesEDA(data_path=data_path)
    return eda.run_full_analysis()

# 병렬 처리
with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    results = list(executor.map(analyze_dataset, data_paths))
```

### 3. 커스텀 시각화

```python
import matplotlib.pyplot as plt

def custom_visualization(df, results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 첫 번째 시점 분포
    axes[0, 0].hist(df.iloc[0].dropna(), bins=50)
    axes[0, 0].set_title('첫 번째 시점 분포')
    
    # 시간별 변화
    axes[0, 1].plot(df.index[:100], df.mean(axis=1)[:100])
    axes[0, 1].set_title('시간별 평균 변화')
    
    # 변수별 표준편차
    axes[1, 0].bar(range(20), df.std().sort_values(ascending=False).head(20))
    axes[1, 0].set_title('상위 20개 변동성 변수')
    
    # 상관관계 히트맵
    corr_matrix = df.corr()
    im = axes[1, 1].imshow(corr_matrix.iloc[:20, :20], cmap='coolwarm')
    axes[1, 1].set_title('상위 20개 변수 상관관계')
    
    plt.colorbar(im, ax=axes[1, 1])
    plt.tight_layout()
    plt.show()
```

## 🐛 문제 해결

### 일반적인 문제들

1. **메모리 부족 오류**
   - 데이터를 청크 단위로 나누어 분석
   - 불필요한 변수 제거
   - 데이터 타입 최적화

2. **시각화 오류**
   - 한글 폰트 설정 확인
   - matplotlib 백엔드 설정
   - 디스플레이 환경 확인

3. **패키지 설치 오류**
   - 가상환경 사용 권장
   - 패키지 버전 호환성 확인
   - 시스템 의존성 설치

### 디버깅 팁

```python
# 디버깅 모드로 실행
import logging
logging.basicConfig(level=logging.DEBUG)

# 메모리 사용량 확인
import psutil
print(f"메모리 사용량: {psutil.virtual_memory().percent}%")

# 데이터 크기 확인
print(f"데이터 크기: {df.shape}")
print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📞 문의

문제가 있거나 개선 사항이 있으시면 이슈를 생성해 주세요.

---

**참고**: 이 도구는 Jupyter Notebook 환경에서도 사용할 수 있으며, 대화형 분석에 최적화되어 있습니다.
