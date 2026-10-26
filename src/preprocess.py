"""Tiền xử lý dạng Pipeline: mọi phép biến đổi chỉ được fit trên dữ liệu huấn luyện."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from data import CAT, NUM


def build_preprocessor(num=NUM, cat=CAT, degree=1):
    """Số: điền thiếu bằng trung vị -> (đa thức bậc `degree`) -> chuẩn hóa. Phân loại: one-hot.

    degree=1 giữ nguyên đặc trưng gốc; bậc cao hơn dùng ở Chương 3.
    """
    numeric = make_pipeline(
        SimpleImputer(strategy="median"),
        PolynomialFeatures(degree, include_bias=False),
        StandardScaler(),
    )
    return ColumnTransformer([("num", numeric, list(num)), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cat))])
