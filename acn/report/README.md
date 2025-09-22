# ACN 정제 공정 종합 분석 시스템

이 프로젝트는 ACN 정제 공정의 효율성 최적화를 위한 종합 분석 시스템입니다. Input_source를 늘려도 높은 수율을 유지할 수 있는 방안을 찾고, 품질값과 Output, Yield 간의 관계를 분석합니다.

## 핵심 문제와 해결 방안

### 기존 문제점
1. **Target을 Yield로 설정**: Input의 영향력이 커서 Input을 줄이는 것만이 답이 됨
2. **과거 값 참조의 한계**: "최적"이라는 결론이 미래 Input 증가 시 어떤 Control 값을 사용해야 하는지 명시하지 않음
3. **품질값 관계 부재**: 품질값과 Output, Yield 간의 관계가 명확하지 않음

### 새로운 접근법
1. **효율성 지표 도입**: Yield 대신 Input 대비 Output의 효율성을 측정하는 종합 지표 사용
2. **미래 지향적 가이드**: Input 증가 시나리오에 대한 명시적인 Control 가이드 제공
3. **품질 통합 분석**: 품질값을 고려한 종합 효율성 점수 계산
4. **실용적 최적화**: 실제 운영에서 활용 가능한 구체적인 방안 제시

## 분석 목표

1. **효율성 최적화**: Input_source 증가 시에도 높은 수율을 유지하는 방안 탐색
2. **품질 관계 분석**: 품질값과 Output, Yield 간의 상관관계 분석
3. **Control 가이드 제공**: 미래 Input 증가에 대한 명시적인 Control 조정 방안
4. **통합 리포트 생성**: 모든 분석 결과를 종합한 HTML 리포트 생성
5. **사용자 맞춤 분석**: 사용자가 직접 텍스트 리포트를 작성할 수 있는 기능 제공

## 파일 구조

```
├── acn_efficiency_optimization.py     # 효율성 최적화 분석 (새로운 핵심 모듈)
├── acn_integrated_report.py           # 통합 HTML 리포트 생성기
├── acn_complete_analysis.py           # 완전 분석 실행 파일
├── acn_exp.py                         # 고급 실험 분석 (기존)
├── control_optimization_analysis.py   # Control 최적화 분석 (기존)
├── acn_*.py                          # 기타 분석 모듈들
└── README.md                         # 프로젝트 설명서
```

## 사용 방법

### 1. 완전 분석 실행 (권장)
```bash
python acn_complete_analysis.py
```
- 샘플 데이터로 시연하거나 실제 데이터 파일로 분석
- 모든 분석을 통합하여 HTML 리포트 생성

### 2. 개별 모듈 실행
```bash
# 효율성 최적화 분석만 실행
python acn_efficiency_optimization.py

# 고급 실험 분석만 실행  
python acn_exp.py

# Control 최적화 분석만 실행
python control_optimization_analysis.py
```

### 3. 통합 리포트 생성
```python
from acn_integrated_report import create_integrated_report

# 모든 분석 결과를 통합하여 HTML 리포트 생성
html_content = create_integrated_report(
    efficiency_optimization_results=efficiency_results,
    advanced_exp_results=advanced_results,
    output_file='my_analysis_report.html'
)
```

## 주요 분석 내용

### 1. 효율성 최적화 분석 (새로운 핵심 기능)
- **효율성 지표 생성**: Yield 대신 Input 대비 Output의 효율성을 측정
- **품질 통합 점수**: 품질값을 고려한 종합 효율성 점수 계산
- **Input-Output 관계 분석**: Input_source와 Output, Yield, Efficiency_Score 간의 상관관계
- **품질값 관계 분석**: 품질값과 Output, Yield 간의 상관관계 분석

### 2. 미래 지향적 Control 가이드
- **Input 증가 시나리오**: 10%, 20%, 30% 증가 시나리오 분석
- **최적 Control 값 계산**: 효율성 유지를 위한 최적 Control 값 자동 계산
- **구체적 조정 방안**: 각 시나리오별 명시적인 Control 조정 가이드 제공

### 3. 고급 예측 모델
- **다양한 모델 비교**: Linear, Ridge, Lasso, Random Forest, SVR
- **성능 평가**: R², RMSE, MAE, Cross-validation 성능
- **특성 중요도**: 각 Control 변수의 중요도 분석

### 4. 통합 시각화
- **Input-Output 관계**: 산점도와 회귀선을 통한 관계 시각화
- **품질값 분포**: 품질값과 각 지표 간의 관계 시각화
- **모델 성능 비교**: 다양한 모델의 성능 비교 차트
- **예측 vs 실제값**: 모델 예측 정확도 시각화

## 주요 발견사항

### 1. 효율성 지표의 우수성
- **Yield 대신 Efficiency_Score 사용**: Input 증가 시에도 효율성 유지 가능
- **품질 통합 효과**: 품질값을 고려한 종합 점수로 더 정확한 최적화
- **미래 지향적 접근**: 과거 데이터에만 의존하지 않는 예측 모델

### 2. Input-Output 관계의 명확성
- **Input_source와 Output**: 강한 양의 상관관계 확인
- **품질값과 Output**: 음의 상관관계로 품질 향상 시 Output 감소 경향
- **효율성 최적화**: Input 증가와 품질 유지의 균형점 발견

### 3. Control 가이드의 실용성
- **구체적 수치 제공**: Input 증가 시나리오별 정확한 Control 값 계산
- **실시간 적용 가능**: 실제 운영에서 즉시 활용 가능한 가이드
- **효율성 유지**: Input 증가에도 불구하고 효율성 점수 유지 가능

## 결론

### 새로운 접근법의 성과

1. **효율성 지표 도입**: Yield 대신 Efficiency_Score를 사용하여 Input 증가 시에도 최적화 가능
2. **미래 지향적 가이드**: Input 증가 시나리오에 대한 구체적인 Control 조정 방안 제공
3. **품질 통합 분석**: 품질값을 고려한 종합적 효율성 평가로 더 정확한 최적화
4. **실용적 해결책**: 실제 운영에서 즉시 활용 가능한 구체적인 수치와 가이드 제공

### 핵심 혁신사항

1. **Target 재정의**: Yield → Efficiency_Score로 변경하여 Input 증가 시나리오 대응
2. **예측 모델 고도화**: 다양한 ML 모델을 통한 정확한 효율성 예측
3. **통합 분석 플랫폼**: 모든 분석 결과를 하나의 HTML 리포트로 통합
4. **사용자 맞춤 기능**: 사용자가 직접 텍스트 리포트를 작성할 수 있는 기능 제공

### 실무 적용 효과

1. **운영 효율성 향상**: Input 증가 시에도 효율성 유지로 생산성 증대
2. **품질 관리 개선**: 품질값과 효율성의 균형점을 통한 최적 운영 조건 도출
3. **의사결정 지원**: 데이터 기반의 구체적이고 실용적인 가이드 제공
4. **지속적 개선**: 통합 리포트를 통한 분석 결과의 체계적 관리

## 필요한 라이브러리

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
SALib (선택사항 - Sobol 분석용)
statsmodels (선택사항 - ANOVA 분석용)
```

## 설치 방법

### 기본 설치
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### 고급 분석 기능 (선택사항)
```bash
pip install SALib statsmodels
```

### 전체 설치
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy SALib statsmodels
```

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다.



