"""Đánh giá mô hình bằng kiểm định chéo trên tập huấn luyện. Các notebook Chương 3 dùng chung để có cùng fold."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from data import SEED

METRICS = {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R2": "r2"}


def cv_folds(train: pd.DataFrame, seed=SEED):
    """5 fold phân tầng theo origin, gom theo name: cùng nguyên tắc với cách tách tập kiểm tra."""
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    return list(cv.split(train, train["origin"], train["name"]))


def cv_score(model, X, y, folds) -> dict:
    """Trung bình và độ lệch chuẩn của RMSE, MAE, R² qua các fold, cho cả phần huấn luyện và phần kiểm định."""
    r = cross_validate(model, X, y, cv=folds, scoring=METRICS, return_train_score=True)
    out = {}
    for k in METRICS:
        for part, ten in (("train", "train"), ("test", "cv")):
            v = r[f"{part}_{k}"] * (1 if k == "R2" else -1)
            out[f"{k}_{ten}"], out[f"{k}_{ten}_std"] = v.mean(), v.std()
    return out


def fmt(mean, std, nd=2) -> str:
    """Định dạng 'trung bình ± độ lệch chuẩn' với dấu phẩy thập phân như trong báo cáo."""
    return f"{mean:.{nd}f} ± {std:.{nd}f}".replace(".", ",")


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
