"""Mức tham chiếu (mô hình cơ sở): CV 5-fold trên tập huấn luyện, không dùng tập kiểm tra.
Chạy từ thư mục gốc dự án: python src/baseline_reference.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from data import CAT, NUM, TARGET, split
from evaluate import cv_folds, cv_score, fmt
from preprocess import build_preprocessor

train, test = split()
X, y = train[NUM + CAT], train[TARGET]
print("train/test:", len(train), len(test))

# CV cũng gom theo name để không có cùng tên xe ở cả phần huấn luyện và phần kiểm định
folds = cv_folds(train)
for name, model in [("Trung bình", DummyRegressor()), ("Hồi quy tuyến tính", LinearRegression())]:
    s = cv_score(make_pipeline(build_preprocessor(), model), X, y, folds)
    print(name, {k: fmt(s[f"{k}_cv"], s[f"{k}_cv_std"]) for k in ("RMSE", "MAE", "R2")})
