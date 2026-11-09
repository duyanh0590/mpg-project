"""Huấn luyện và đánh giá Polynomial Regression và Ridge Regression cho ứng dụng AutoMPG Analytics.

File này không dùng Streamlit, chỉ có pandas/scikit-learn, nên có thể chạy và kiểm tra riêng:
    python app/modeling.py
Dùng chung cách chia tập (src/data.py), tiền xử lý (src/preprocess.py) và 5 fold CV (src/evaluate.py)
với các notebook, nên số liệu khớp với báo cáo.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline

from data import CAT, TARGET, load_raw, split
from evaluate import cv_folds, rmse
from preprocess import build_preprocessor

# Bậc rất cao làm ma trận gần suy biến; scikit-learn cảnh báo nhưng vẫn cho kết quả (và kết quả đó được hiển thị).
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Nhánh đa thức dùng 4 đặc trưng như notebook 03 và báo cáo (mục 3.2): cylinders và displacement
# tương quan 0,90–0,95 với weight/horsepower nên không đưa vào để số cột không tăng vô ích.
POLY = ["weight", "horsepower", "acceleration", "model_year"]
FEATURES = POLY + CAT
DEGREES = list(range(1, 11))
ALPHAS = np.logspace(-4, 4, 17)          # 10^-4 ... 10^4, mỗi bước nửa bậc 10
ORIGIN_VI = {"usa": "Mỹ", "europe": "Châu Âu", "japan": "Nhật Bản"}
LABELS = {
    "mpg": "MPG", "cylinders": "Cylinders", "displacement": "Displacement (in³)", "horsepower": "Horsepower (hp)",
    "weight": "Weight (lb)", "acceleration": "Acceleration (giây 0–60 mph)", "model_year": "Model Year", "origin": "Origin",
}


def load_data():
    """Trả về (toàn bộ dữ liệu, tập huấn luyện, tập kiểm thử)."""
    df = load_raw()
    train, test = split(df)
    return df, train, test


def make_model(degree, alpha=None):
    """Pipeline: điền thiếu (trung vị) -> PolynomialFeatures(degree) -> StandardScaler -> hồi quy.
    alpha=None: Polynomial Regression (bình phương tối thiểu); alpha>0: Ridge Regression."""
    reg = LinearRegression() if alpha is None else Ridge(alpha=alpha)
    return make_pipeline(build_preprocessor(POLY, CAT, degree), reg)


def scores(model, train, test):
    """Khớp mô hình trên tập huấn luyện rồi tính chỉ số trên cả hai tập."""
    model.fit(train[FEATURES], train[TARGET])
    p_tr, p_te = model.predict(train[FEATURES]), model.predict(test[FEATURES])
    return {
        "Train R²": r2_score(train[TARGET], p_tr), "Test R²": r2_score(test[TARGET], p_te),
        "Train RMSE": rmse(train[TARGET], p_tr), "Test RMSE": rmse(test[TARGET], p_te),
        "Train MAE": mean_absolute_error(train[TARGET], p_tr), "Test MAE": mean_absolute_error(test[TARGET], p_te),
    }


def _cv_rmse_by_alpha(train, degree, alphas):
    """RMSE CV 5-fold cho mọi alpha (alpha=None: không điều chuẩn) ở một bậc.
    Tiền xử lý không phụ thuộc alpha nên chỉ khớp một lần mỗi fold rồi thử lần lượt các alpha (nhanh hơn ~15 lần)."""
    X, y = train[FEATURES], train[TARGET].to_numpy()
    out = [[] for _ in alphas]
    for tr, va in cv_folds(train):
        pre = build_preprocessor(POLY, CAT, degree).fit(X.iloc[tr])
        Ztr, Zva = pre.transform(X.iloc[tr]), pre.transform(X.iloc[va])
        for i, a in enumerate(alphas):
            reg = LinearRegression() if a is None else Ridge(alpha=a)
            out[i].append(rmse(y[va], reg.fit(Ztr, y[tr]).predict(Zva)))
    return [(float(np.mean(v)), float(np.std(v))) for v in out]


def polynomial_table(train, test):
    """Bảng kết quả Polynomial Regression bậc 1–10."""
    rows = []
    for d in DEGREES:
        m = make_model(d)
        s = scores(m, train, test)
        (cv, cv_std), = _cv_rmse_by_alpha(train, d, [None])
        rows.append({"Degree": d, "Số đặc trưng": m[0].transform(train[FEATURES].iloc[:1]).shape[1],
                     **s, "CV RMSE": cv, "CV RMSE std": cv_std})
    return pd.DataFrame(rows).set_index("Degree")


def ridge_table(train, test):
    """Bảng Ridge cho mọi cặp (bậc, alpha): chỉ số train/test, RMSE CV và độ lớn vector hệ số ||w||."""
    rows = []
    y_tr, y_te = train[TARGET], test[TARGET]
    for d in DEGREES:
        cv = _cv_rmse_by_alpha(train, d, list(ALPHAS))
        pre = build_preprocessor(POLY, CAT, d).fit(train[FEATURES])     # giống make_model(d, a) nhưng khớp tiền xử lý một lần
        Ztr, Zte = pre.transform(train[FEATURES]), pre.transform(test[FEATURES])
        for a, (c, c_std) in zip(ALPHAS, cv):
            reg = Ridge(alpha=a).fit(Ztr, y_tr)
            p_tr, p_te = reg.predict(Ztr), reg.predict(Zte)
            rows.append({"Degree": d, "alpha": a,
                         "Train R²": r2_score(y_tr, p_tr), "Test R²": r2_score(y_te, p_te),
                         "Train RMSE": rmse(y_tr, p_tr), "Test RMSE": rmse(y_te, p_te),
                         "Train MAE": mean_absolute_error(y_tr, p_tr), "Test MAE": mean_absolute_error(y_te, p_te),
                         "CV RMSE": c, "CV RMSE std": c_std, "||w||": float(np.linalg.norm(reg.coef_))})
    return pd.DataFrame(rows)


def best_ridge_per_degree(ridge):
    """Với mỗi bậc, chọn alpha có RMSE CV thấp nhất (chọn bằng CV, không nhìn tập kiểm thử)."""
    return ridge.loc[ridge.groupby("Degree")["CV RMSE"].idxmin()].set_index("Degree")


def candidates(poly, ridge):
    """Danh sách mô hình ứng viên cho trang Tổng quan: Polynomial bậc 1–10 và Ridge (alpha tốt nhất theo CV) bậc 1–10."""
    p = poly.assign(Model=[f"Polynomial bậc {d}" for d in poly.index], alpha=np.nan)
    r = best_ridge_per_degree(ridge)
    r = r.assign(Model=[f"Ridge bậc {d}, α = {a:.3g}" for d, a in zip(r.index, r["alpha"])])
    cols = ["Model", "alpha", "Train R²", "Test R²", "Test RMSE", "Test MAE", "CV RMSE"]
    return pd.concat([p.reset_index()[["Degree"] + cols], r.reset_index()[["Degree"] + cols]], ignore_index=True)


def car(cylinders, displacement, horsepower, weight, acceleration, model_year, origin):
    """Một dòng dữ liệu đúng định dạng bộ Auto MPG để đưa vào mô hình."""
    return pd.DataFrame([dict(cylinders=cylinders, displacement=displacement, horsepower=horsepower, weight=weight,
                              acceleration=acceleration, model_year=model_year, origin=origin)])


def weight_curve(train, degree, n=200):
    """Minh họa 1 chiều: hồi quy đa thức MPG theo riêng weight, trả về điểm vẽ đường cong."""
    m = make_pipeline(build_preprocessor(["weight"], [], degree), LinearRegression())
    m.fit(train[["weight"]], train[TARGET])
    w = np.linspace(train["weight"].min(), train["weight"].max(), n)
    return w, m.predict(pd.DataFrame({"weight": w}))


if __name__ == "__main__":
    import time
    t = time.time()
    df, train, test = load_data()
    poly = polynomial_table(train, test)
    print(poly[["Số đặc trưng", "Train R²", "Test R²", "Test RMSE", "CV RMSE"]].round(3))
    ridge = ridge_table(train, test)
    print(best_ridge_per_degree(ridge)[["alpha", "Test RMSE", "CV RMSE"]].round(3))
    # Phải khớp notebook 03: đa thức bậc 2 RMSE CV 2,87; Ridge bậc 6 alpha = 10 RMSE CV 2,83
    assert abs(poly.loc[2, "CV RMSE"] - 2.87) < 0.01
    print(f"Xong sau {time.time() - t:.1f} giây")
