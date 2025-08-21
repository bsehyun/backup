# CSV 추론 시스템

CSV 파일과 목표 시간을 입력받아 추론을 수행하고 결과를 CSV에 추가하는 시스템입니다.

## 주요 기능

- **입력**: CSV 파일 (시간 인덱스, 피처 컬럼) + 목표 시간
- **출력**: 결과 CSV 파일 (목표시간, 추론값, 비고 컬럼)
- **특징**: 
  - 방어적 프로그래밍으로 예외처리 완비
  - 연산 비용 최적화 (필요시에만 모델 로드)
  - 상세한 오류 메시지 제공
  - 기존 결과에 새로운 행 추가

## 시스템 요구사항

### 사전 준비된 모델
- `short_model.pkl`: 단기 모델 (Gradient Boosting)
- `long_model.pkl`: 장기 모델 (Gradient Boosting)
- `short_scaler.pkl`: 단기 모델용 스케일러 (RobustScaler)
- `long_scaler.pkl`: 장기 모델용 스케일러 (RobustScaler)

### 피처 변형 규칙
모델의 피처 이름에 따라 자동으로 변형됩니다:

1. **원천 피처**: `{원천feature이름}`
2. **지연 피처**: `{원천feature이름}_{lag}_{시간}m`
3. **변화율 피처**: `{원천feature이름}_{roc}_{시간}m_{median or std}`

### 최종 예측 계산
```python
final_prediction = short_model_pred + long_model_pred
```

### 상태 판정
- 추론값 ≥ 68000: "경고"
- 추론값 < 68000: "정상"
- 오류 발생: "추론제한"

## 파일 구조

```
├── inference_system.py              # 클래스 기반 구현
├── inference_system_functional.py   # 함수형 구현
├── inference_notebook.py            # Jupyter Notebook용
├── example_usage.py                 # 사용 예시 및 테스트
├── README.md                        # 이 파일
└── models/                          # 모델 디렉토리
    ├── short_model.pkl
    ├── long_model.pkl
    ├── short_scaler.pkl
    └── long_scaler.pkl
```

## 사용법

### 1. 기본 사용법

```python
from inference_notebook import run_inference

# 현재 시간으로 추론
result = run_inference('input_data.csv')

# 특정 시간으로 추론
result = run_inference('input_data.csv', '2024-01-15 14:30:00')

print(result)
```

### 2. 직접 함수 호출

```python
from inference_notebook import predict

result = predict('input_data.csv', '2024-01-15 14:30:00', './models')
print(result)
```

### 3. Jupyter Notebook에서 사용

```python
# 셀 1: 라이브러리 import
from inference_notebook import *

# 셀 2: 추론 실행
result = run_inference('my_data.csv', '2024-01-15 14:30:00')
print(result)
```

## 입력 CSV 형식

```csv
timestamp,feature1,feature2,feature3,feature4,feature5
2024-01-15 12:00:00,100.5,50.2,200.1,75.3,150.7
2024-01-15 12:01:00,101.2,49.8,201.5,76.1,149.9
...
```

**요구사항:**
- 인덱스가 시간 형식이어야 함
- 최소 60개 행 필요
- 시간이 오름차순으로 정렬되어야 함

## 출력 CSV 형식

```csv
목표시간,추론값,비고
2024-01-15 14:30:00,67500.25,정상
2024-01-15 15:00:00,68500.50,경고
```

## 예외 처리

시스템은 다음과 같은 예외 상황을 처리합니다:

1. **파일 관련 오류**
   - 존재하지 않는 CSV 파일
   - CSV 파일 읽기 실패

2. **데이터 검증 오류**
   - 빈 CSV 파일
   - 시간 인덱스 없음
   - 유효하지 않은 시간 형식
   - 데이터 길이 부족 (60개 미만)
   - 시간 순서 오류

3. **피처 관련 오류**
   - 필요한 피처 누락
   - 피처 변형 실패

4. **모델 관련 오류**
   - 모델 파일 없음
   - 모델 로드 실패
   - 예측 실패

5. **결과 저장 오류**
   - 결과 파일 쓰기 실패

## 테스트 실행

```bash
python example_usage.py
```

이 명령어는 다음을 수행합니다:
1. 샘플 데이터 생성
2. 샘플 모델 생성
3. 기본 기능 테스트
4. 오류 처리 테스트

## 성능 최적화

1. **지연 로딩**: 모델은 필요시에만 로드됩니다.
2. **메모리 효율성**: 전역 변수로 모델을 한 번만 로드합니다.
3. **연산 최적화**: 피처 변형은 필요한 경우에만 수행됩니다.

## 주의사항

1. **모델 파일**: `./models/` 디렉토리에 필요한 모델 파일들이 있어야 합니다.
2. **피처 이름**: 모델의 피처 이름과 CSV의 컬럼 이름이 일치해야 합니다.
3. **시간 형식**: CSV의 인덱스는 pandas가 파싱할 수 있는 시간 형식이어야 합니다.
4. **데이터 길이**: 최소 60개 행의 데이터가 필요합니다.

## 문제 해결

### 일반적인 오류

1. **"모델이 로드되지 않았습니다"**
   - `./models/` 디렉토리에 모델 파일들이 있는지 확인
   - 파일 권한 확인

2. **"필요한 피처가 누락되었습니다"**
   - CSV 파일에 필요한 컬럼들이 있는지 확인
   - 컬럼 이름이 정확한지 확인

3. **"데이터가 충분하지 않습니다"**
   - CSV 파일에 최소 60개 행이 있는지 확인

4. **"인덱스가 유효한 시간 형식이 아닙니다"**
   - CSV의 인덱스가 시간 형식인지 확인
   - `index_col=0`으로 설정되어 있는지 확인

### 로그 확인

시스템은 상세한 로그를 출력하므로, 오류 발생 시 로그 메시지를 확인하여 문제를 파악할 수 있습니다.
