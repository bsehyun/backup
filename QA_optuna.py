import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score

def tune_with_optuna(df: pd.DataFrame, y_true: np.ndarray, n_trials=200):
    crop_cols = [c for c in df.columns if "crop" in c.lower()]
    full_cols = [c for c in df.columns if "full" in c.lower()]
    
    def objective(trial):
        # 어떤 조합 방식을 쓸지 결정
        logic = trial.suggest_categorical("logic", ["single", "or", "and"])
        
        if logic == "single":
            # crop / full 상관없이 단일 모델 선택
            col = trial.suggest_categorical("model", df.columns.tolist())
            th = trial.suggest_float("threshold", 0.0, 1.0)
            y_pred = (df[col].values >= th).astype(int)
            
        else:
            crop_col = trial.suggest_categorical("crop_model", crop_cols)
            full_col = trial.suggest_categorical("full_model", full_cols)
            th_c = trial.suggest_float("th_crop", 0.0, 1.0)
            th_f = trial.suggest_float("th_full", 0.0, 1.0)
            
            crop_pred = (df[crop_col].values >= th_c).astype(int)
            full_pred = (df[full_col].values >= th_f).astype(int)
            
            if logic == "or":
                y_pred = np.logical_or(crop_pred, full_pred).astype(int)
            else:  # "and"
                y_pred = np.logical_and(crop_pred, full_pred).astype(int)
        
        score = f1_score(y_true, y_pred)
        return score
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    return study
