"""Mức tham chiếu (mô hình cơ sở): CV 5-fold trên tập huấn luyện, không dùng tập kiểm tra.
Chạy từ thư mục gốc dự án: python src/baseline_reference.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.pipeline import make_pipeline

from data import CAT, NUM, SEED, TARGET, split
from preprocess import build_preprocessor

train, test = split()
X, y, groups = train[NUM + CAT], train[TARGET], train["name"]
print("train/test:", len(train), len(test))

# CV cũng gom theo name để không có cùng tên xe ở cả phần huấn luyện và phần kiểm định
cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
folds = list(cv.split(X, train["origin"], groups))
metrics = {"RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error", "R2": "r2"}
for name, model in [("Trung bình", DummyRegressor()), ("Hồi quy tuyến tính", LinearRegression())]:
    r = cross_validate(make_pipeline(build_preprocessor(), model), X, y, cv=folds, scoring=metrics)
    print(name, {k: f"{abs(r['test_' + k].mean()):.2f} ± {r['test_' + k].std():.2f}" for k in metrics})
