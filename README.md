# 추론 시스템 (Inference System) 사용 매뉴얼

## 개요

이 시스템은 CSV 파일과 목표 시간을 입력받아 머신러닝 모델을 사용한 추론을 수행하고, 결과를 CSV 파일에 저장하는 도구입니다.

## 주요 기능

- 📊 CSV 데이터 파일 읽기 및 검증
- 🤖 단기/장기 모델을 통한 예측 수행
- 📈 피처 엔지니어링 (지연 피처, 변화율 피처 등)
- 💾 결과를 CSV 파일로 자동 저장
- ⚠️ 예측값에 따른 상태 판단 (정상/경고)

## 설치 및 준비

### 1. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 모델 파일 준비

`./models/` 디렉토리에 다음 파일들이 필요합니다:
- `short_model.pkl` - 단기 예측 모델
- `long_model.pkl` - 장기 예측 모델
- `short_scaler.pkl` - 단기 모델용 스케일러
- `long_scaler.pkl` - 장기 모델용 스케일러

## 사용 방법

### 기본 사용법

```python
from inference_system import run_inference

# 기본 사용 (현재 시간 기준)
result = run_inference("input_data.csv")
print(result)

# 특정 시간 지정
result = run_inference("input_data.csv", "2024-01-15 14:30:00")
print(result)

# 다른 모델 디렉토리 사용
result = run_inference("input_data.csv", model_dir="./custom_models")
print(result)
```

### 고급 사용법 (클래스 직접 사용)

```python
from inference_system import InferenceSystem

# 시스템 초기화
system = InferenceSystem(model_dir="./models")

# 추론 수행
result = system.predict("input_data.csv", "2024-01-15 14:30:00")
print(result)
```

## 입력 CSV 파일 형식

### 필수 요구사항

1. **시간 인덱스**: 첫 번째 열이 시간으로 설정되어야 함
2. **최소 데이터**: 최소 60개 행 (1시간 분량) 필요
3. **시간 순서**: 오름차순으로 정렬되어야 함
4. **필요한 피처**: 모델이 요구하는 피처들이 포함되어야 함

### 예시 CSV 형식

```csv
timestamp,feature1,feature2,feature3,feature4,feature5
2024-01-15 13:00:00,100.5,200.3,50.1,75.2,120.8
2024-01-15 13:01:00,101.2,201.1,50.5,75.8,121.2
2024-01-15 13:02:00,100.8,200.7,50.3,75.5,120.9
...
```

### 지원하는 피처 형식

시스템은 다음 형식의 피처를 자동으로 생성합니다:

1. **원본 피처**: `feature1`, `feature2` 등
2. **지연 피처**: `feature1_5_10m` (5단계 지연, 10분 윈도우)
3. **변화율 피처**: `feature1_roc_3_15m_median` (3기간 변화율, 15분 윈도우, 중앙값)

## 출력 결과

### 1. 콘솔 출력

```
추론 완료 - 최종값: 67500.50, 상태: 정상
```

### 2. 결과 CSV 파일

`input_data_results.csv` 파일이 생성되며, 다음 형식으로 저장됩니다:

```csv
목표시간,추론값,비고
2024-01-15 14:30:00,67500.50,정상
```

### 상태 판단 기준

- **정상**: 추론값 < 68,000
- **경고**: 추론값 ≥ 68,000

## 오류 처리

### 일반적인 오류 메시지

| 오류 메시지 | 원인 | 해결 방법 |
|------------|------|----------|
| "입력 CSV가 비어있습니다." | CSV 파일이 비어있음 | 데이터가 포함된 CSV 파일 사용 |
| "CSV의 인덱스가 시간으로 설정되어 있지 않습니다." | 시간 인덱스 누락 | 첫 번째 열을 시간으로 설정 |
| "데이터가 충분하지 않습니다." | 최소 60개 행 미만 | 더 많은 데이터 제공 |
| "시간이 오름차순으로 정렬되어 있지 않습니다." | 시간 순서 오류 | 시간순으로 정렬 |
| "필요한 피처가 누락되었습니다." | 모델이 요구하는 피처 부족 | 필요한 피처 추가 |
| "필요한 모델이 로드되지 않았습니다." | 모델 파일 누락 | 모델 파일 확인 |

## 사용 예시

### 예시 1: 기본 추론

```python
# 현재 시간 기준으로 추론
result = run_inference("sensor_data.csv")
print(result)
```

### 예시 2: 특정 시간 추론

```python
# 특정 시간에 대한 추론
result = run_inference("sensor_data.csv", "2024-01-15 14:30:00")
print(result)
```

### 예시 3: 배치 처리

```python
import pandas as pd
from inference_system import InferenceSystem

# 시스템 초기화
system = InferenceSystem()

# 여러 파일 처리
files = ["data1.csv", "data2.csv", "data3.csv"]
times = ["2024-01-15 14:30:00", "2024-01-15 15:00:00", "2024-01-15 15:30:00"]

for file, time in zip(files, times):
    result = system.predict(file, time)
    print(f"{file}: {result}")
```

## 성능 고려사항

### 메모리 사용량
- 대용량 CSV 파일 처리 시 메모리 사용량 증가
- 피처 생성 과정에서 임시 데이터 생성

### 처리 시간
- 피처 엔지니어링: 데이터 크기에 비례
- 모델 예측: 거의 즉시 처리
- 파일 I/O: CSV 크기에 비례

### 권장사항
- 대용량 데이터 처리 시 청크 단위로 분할 고려
- 정기적인 모델 업데이트 권장

## 문제 해결

### 자주 발생하는 문제

1. **모델 로드 실패**
   ```python
   # 모델 디렉토리 확인
   import os
   print(os.listdir("./models"))
   ```

2. **피처 누락 오류**
   ```python
   # CSV 파일의 컬럼 확인
   import pandas as pd
   df = pd.read_csv("input_data.csv")
   print(df.columns.tolist())
   ```

3. **시간 형식 오류**
   ```python
   # 시간 인덱스 확인
   df = pd.read_csv("input_data.csv", index_col=0)
   print(df.index[:5])
   ```

## 지원 및 문의

문제가 발생하거나 추가 기능이 필요한 경우:
1. 오류 메시지를 정확히 기록
2. 입력 데이터 형식 확인
3. 모델 파일 존재 여부 확인

## 라이선스

이 프로젝트는 내부 사용을 위한 도구입니다. 