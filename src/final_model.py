"""Mô hình cuối (Voting: Ridge + Gradient Boosting) với siêu tham số đã chọn ở notebooks/05_improvement.ipynb.
Chạy từ thư mục gốc dự án để kiểm tra lại kết quả trên tập kiểm tra: python src/final_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from data import CAT, NUM, SEED, TARGET, split
from features import DIESEL, LOG, RATIO, add_features
from preprocess import build_preprocessor

POLY = ["weight", "horsepower", "acceleration", "model_year"]


def build_final_model():
    """Cấu hình lấy từ GridSearchCV ở notebook 05 (Bảng 19 trong báo cáo)."""
    ridge = TransformedTargetRegressor(make_pipeline(
        FunctionTransformer(add_features), build_preprocessor(POLY + DIESEL, CAT, degree=3), Ridge(alpha=10 ** -2.5)))
    gb = TransformedTargetRegressor(make_pipeline(
        FunctionTransformer(add_features), build_preprocessor(NUM + LOG + RATIO + DIESEL, CAT, degree=1),
        GradientBoostingRegressor(n_estimators=600, learning_rate=0.03, max_depth=2, subsample=1.0, random_state=SEED)),
        func=np.log, inverse_func=np.exp)
    return VotingRegressor([("ridge", ridge), ("gb", gb)])


def train_final_model():
    """Khớp mô hình cuối trên toàn bộ 318 mẫu huấn luyện (như mục 3.6.2)."""
    train, _ = split()
    return build_final_model().fit(train, train[TARGET])


def car(cylinders, displacement, horsepower, weight, acceleration, model_year, origin, diesel=False, name="xe nhap tay"):
    """Một dòng dữ liệu cho mô hình. Cờ diesel được mô hình đọc từ tên xe (src/features.py)."""
    return pd.DataFrame([{"cylinders": cylinders, "displacement": displacement, "horsepower": horsepower,
                          "weight": weight, "acceleration": acceleration, "model_year": model_year,
                          "origin": origin, "name": name + (" diesel" if diesel else "")}])


if __name__ == "__main__":
    train, test = split()
    p = build_final_model().fit(train, train[TARGET]).predict(test)
    rmse = float(np.sqrt(np.mean((test[TARGET] - p) ** 2)))
    print(f"RMSE trên tập kiểm tra: {rmse:.3f} (báo cáo: 2,314)")
    assert abs(rmse - 2.314) < 0.001, "kết quả không khớp notebook 05"
