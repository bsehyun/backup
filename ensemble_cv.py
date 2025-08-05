import numpy as np
import optuna
from sklearn.metrics import mean_squared_error

# ==== 예시 데이터 (이미 있는 걸로 대체하세요) ====
# 시계열 순서대로 정렬된 상태여야 함
# A_pred_all, B_pred_all: 각 모델의 예측값
# y_all: 정답값
# 예: 길이가 N인 numpy array
# -----------------------------------------------
# A_pred_all = np.array([...])
# B_pred_all = np.array([...])
# y_all = np.array([...])

def timeseries_folds(A_pred, B_pred, y, n_val=1):
    """
    Walk-forward CV fold 생성기
    n_val: 각 fold의 validation 데이터 개수
    """
    N = len(y)
    for start_val in range(len(y) - n_val, 0, -n_val):
        train_idx = np.arange(0, start_val)
        val_idx = np.arange(start_val, start_val + n_val)
        if len(val_idx) == 0:
            continue
        yield train_idx, val_idx

def objective(trial, A_pred, B_pred, y):
    # 튜닝 파라미터
    threshold = trial.suggest_float("threshold", min(A_pred), max(A_pred))
    alpha = trial.suggest_float("alpha", 0.0, 1.0)  # threshold 초과 시 blending
    beta = trial.suggest_float("beta", 0.0, 1.0)    # threshold 이하 시 blending

    train_scores = []
    val_scores = []

    # Inner CV (time-series walk-forward)
    for train_idx, val_idx in timeseries_folds(A_pred, B_pred, y, n_val=2):
        A_train, B_train, y_train = A_pred[train_idx], B_pred[train_idx], y[train_idx]
        A_val, B_val, y_val = A_pred[val_idx], B_pred[val_idx], y[val_idx]

        # Train 예측
        pred_train = np.where(
            A_train > threshold,
            alpha * A_train + (1 - alpha) * B_train,
            beta * A_train + (1 - beta) * B_train
        )
        # Val 예측
        pred_val = np.where(
            A_val > threshold,
            alpha * A_val + (1 - alpha) * B_val,
            beta * A_val + (1 - beta) * B_val
        )

        train_rmse = mean_squared_error(y_train, pred_train, squared=False)
        val_rmse = mean_squared_error(y_val, pred_val, squared=False)

        train_scores.append(train_rmse)
        val_scores.append(val_rmse)

    # Train/Val 가중 평균 RMSE (낮을수록 좋음)
    score = 0.3 * np.mean(train_scores) + 0.7 * np.mean(val_scores)
    return score

# ==== Outer CV (Nested CV) ====
def nested_cv_optuna(A_pred_all, B_pred_all, y_all, n_outer=3):
    N = len(y_all)
    fold_size = N // n_outer
    outer_results = []

    for outer_fold in range(n_outer):
        # Outer Test index
        test_start = outer_fold * fold_size
        test_end = test_start + fold_size
        test_idx = np.arange(test_start, test_end)
        
        # Outer Train/Val index
        trainval_idx = np.setdiff1d(np.arange(N), test_idx)

        A_trainval, B_trainval, y_trainval = A_pred_all[trainval_idx], B_pred_all[trainval_idx], y_all[trainval_idx]
        A_test, B_test, y_test = A_pred_all[test_idx], B_pred_all[test_idx], y_all[test_idx]

        # Optuna 튜닝
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda t: objective(t, A_trainval, B_trainval, y_trainval), n_trials=50)

        best_params = study.best_params

        # Outer Test 평가
        pred_test = np.where(
            A_test > best_params["threshold"],
            best_params["alpha"] * A_test + (1 - best_params["alpha"]) * B_test,
            best_params["beta"] * A_test + (1 - best_params["beta"]) * B_test
        )
        test_rmse = mean_squared_error(y_test, pred_test, squared=False)

        outer_results.append({
            "outer_fold": outer_fold,
            "best_params": best_params,
            "test_rmse": test_rmse
        })

    return outer_results

# 실행 예시
# results = nested_cv_optuna(A_pred_all, B_pred_all, y_all, n_outer=3)
# print(results)
