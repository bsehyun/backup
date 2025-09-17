# Control 최적화 분석

이 프로젝트는 Control 변수와 Target 변수 간의 관계를 분석하고, 왜 특정 Control 값들(95, 94)에서 예측 성능이 떨어지는지 증명하는 코드입니다.

## 문제 상황

- Control 변수는 ordinal continuous하지만 거의 classified된 선택지
- Control → Target 회귀 예측 모델이 이미 구축됨
- Control 96, 97, 98에 대한 예측은 잘 작동하지만, 95, 94 예측은 이상한 결과
- Non-linear regressor 사용 시 발생하는 문제
- Linear 모델로는 학습이 잘 되지 않음

## 분석 목표

1. **데이터 분포 및 특성 분석**: Control별 샘플 수와 Target 분포 파악
2. **샘플 불균형 문제 식별**: 데이터 불균형이 예측 성능에 미치는 영향 분석
3. **선형 vs 비선형 모델 성능 비교**: 다양한 모델의 성능 차이 분석
4. **Control별 예측 오차 패턴 분석**: 특정 Control 값에서 오차가 큰 이유 분석
5. **종합 분석 리포트 생성**: 근거 기반 결론 도출

## 파일 구조

```
├── control_optimization_analysis.py    # Python 스크립트 버전
├── control_optimization_analysis.ipynb # Jupyter Notebook 버전
└── README.md                          # 프로젝트 설명서
```

## 사용 방법

### Python 스크립트 실행
```bash
python control_optimization_analysis.py
```

### Jupyter Notebook 실행
```bash
jupyter notebook control_optimization_analysis.ipynb
```

## 주요 분석 내용

### 1. 데이터 특성
- **합성 데이터 생성**: 실제 상황을 모방한 데이터
  - Control 96, 97, 98: 각각 100개 샘플, 선형적 관계
  - Control 95, 94: 각각 15개 샘플, 비선형적 관계

### 2. 시각화
- Control별 Target 분포 (Box plot, Violin plot)
- 샘플 수 분포
- Control vs Target 산점도
- 예측 오차 패턴 분석

### 3. 모델 비교
- **Linear Regression**: 단순한 선형 관계 모델링
- **Random Forest**: 앙상블 기반 비선형 모델
- **SVR (RBF)**: 서포트 벡터 회귀 (RBF 커널)

### 4. 성능 지표
- MSE (Mean Squared Error)
- R² (결정계수)
- MAE (Mean Absolute Error)
- Cross-validation 성능

## 주요 발견사항

### 1. 샘플 불균형 문제
- Control 96, 97, 98: 충분한 샘플 (각각 100개)
- Control 95, 94: 부족한 샘플 (각각 15개)
- 불균형 비율: 약 6.67:1

### 2. 모델 성능 차이
- **비선형 모델**: 전체적으로 높은 R² 값, 하지만 과적합 위험
- **선형 모델**: 낮은 R² 값, 하지만 안정적인 예측

### 3. Control별 예측 오차 패턴
- **충분한 샘플 (96, 97, 98)**: 낮은 예측 오차
- **부족한 샘플 (95, 94)**: 높은 예측 오차

## 결론

### 왜 Control 95, 94에서 예측 성능이 떨어지는가?

1. **샘플 불균형**: Control 95, 94에 대한 샘플 수가 매우 적음 (각각 15개)
2. **비선형 모델의 과적합**: 적은 샘플에 대해 복잡한 패턴을 학습하여 일반화 능력 저하
3. **데이터 분포 차이**: Control 95, 94는 다른 Control 값들과 다른 분포 특성
4. **노이즈 영향**: 적은 샘플로 인해 노이즈의 영향이 상대적으로 큼

### 개선 방향

1. **데이터 수집**: Control 95, 94에 대한 더 많은 샘플 확보
2. **정규화 기법**: L1/L2 정규화를 통한 과적합 방지
3. **앙상블 방법**: 여러 모델의 예측 결과 결합
4. **도메인 지식 활용**: Control-Target 관계에 대한 사전 지식 반영

## 필요한 라이브러리

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
```

## 설치 방법

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.



