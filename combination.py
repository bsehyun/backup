import numpy as np
from scipy.optimize import minimize

# ✅ 샘플 데이터 생성 (10개 샘플, 6개 모델)
np.random.seed(42)
probs = np.random.rand(10, 6)  # 모델별 확률값 (10개 샘플, 6개 모델)
y_true = np.random.randint(0, 2, size=10)  # Ground Truth (1: Positive, 0: Negative)

# ✅ 평가 함수: Recall=1을 만족하는 최소 threshold와 weight 찾기
def objective(params):
    weights = params[:6]  # 가중치 (6개 모델)
    threshold = params[6]  # 임곗값 (threshold)
    
    # 가중 합 적용
    weighted_sum = np.dot(probs, weights)
    
    # 예측값 생성
    predictions = (weighted_sum >= threshold).astype(int)
    
    # Recall 계산
    true_positives = np.sum((predictions == 1) & (y_true == 1))
    actual_positives = np.sum(y_true)
    
    recall = true_positives / actual_positives if actual_positives > 0 else 1

    # Recall = 1을 만족하는 가중치와 threshold를 찾기
    if recall < 1:
        return 1e6  # Recall이 1 미만이면 큰 패널티 부여 (최적화 배제)
    
    # threshold 최소화 (Recall=1을 만족하는 가장 작은 threshold 찾기)
    return threshold

# ✅ 초기값 (가중치는 균등, threshold는 0.5)
initial_params = np.append(np.ones(6) / 6, 0.5)

# ✅ 가중치의 합이 1이 되도록 제약 조건 설정
constraints = [{"type": "eq", "fun": lambda x: np.sum(x[:6]) - 1}]  # 가중치 합 = 1

# ✅ 가중치 범위 (0~1) & threshold 범위 (0~1)
bounds = [(0, 1)] * 6 + [(0, 1)]

# ✅ 최적화 실행
result = minimize(objective, initial_params, bounds=bounds, constraints=constraints, method="SLSQP")

# ✅ 최적 가중치와 threshold 출력
optimal_weights = result.x[:6]
optimal_threshold = result.x[6]

print(f"최적 가중치: {optimal_weights}")
print(f"최적 threshold: {optimal_threshold:.4f}")













import numpy as np
import pandas as pd
from itertools import product
from scipy.optimize import minimize

# 🔹 예제 데이터 (샘플 10개, 모델 6개)
np.random.seed(42)
probs = np.random.rand(10, 6)  # (10개 샘플, 6개 모델)

# 🔹 Ground Truth (1: Positive, 0: Negative)
y_true = np.random.randint(0, 2, size=10)

# ✅ 1. 조합 방법 정의
prob_methods = {
    "max": np.max(probs, axis=1),
    "mean": np.mean(probs, axis=1),
    "product": np.prod(probs, axis=1),
    "sum": np.sum(probs, axis=1),
}

# 가중 평균 (기본 가중치 동일)
weights = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])  # 가중합 (sum이 1)
prob_methods["weighted_sum"] = np.dot(probs, weights)

# ✅ 2. Threshold 최적화 (각 기법별로 recall=1 되는 최소 threshold 찾기)
best_thresholds = {}
for method, combined_prob in prob_methods.items():
    possible_thresholds = np.linspace(0, 1, 100)  # 0~1 사이 100개 탐색

    for threshold in possible_thresholds:
        predictions = (combined_prob >= threshold).astype(int)
        recall = np.sum((predictions == 1) & (y_true == 1)) / max(1, np.sum(y_true))  # recall 계산

        if recall == 1:  # Recall=1이 되는 최소 threshold 찾기
            best_thresholds[method] = threshold
            break

# ✅ 3. 최적 threshold 출력
for method, threshold in best_thresholds.items():
    print(f"{method}: 최적 threshold = {threshold:.4f}")
