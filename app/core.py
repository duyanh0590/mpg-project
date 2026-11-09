"""Dữ liệu và mô hình được tính một lần rồi lưu cache, mọi trang dùng chung."""
import streamlit as st

import modeling as M


@st.cache_data(show_spinner=False)
def data():
    """(toàn bộ dữ liệu, tập huấn luyện, tập kiểm thử)"""
    return M.load_data()


@st.cache_data(show_spinner="Đang huấn luyện Polynomial Regression bậc 1–10…")
def poly_results():
    _, train, test = data()
    return M.polynomial_table(train, test)


@st.cache_data(show_spinner="Đang huấn luyện Ridge Regression (10 bậc × 17 giá trị α, lần đầu khoảng 10 giây)…")
def ridge_results():
    _, train, test = data()
    return M.ridge_table(train, test)


@st.cache_resource(show_spinner=False)
def fitted_model(degree, alpha=None):
    """Mô hình đã khớp trên tập huấn luyện, dùng để dự đoán ở trang Dự đoán MPG."""
    _, train, _ = data()
    return M.make_model(degree, alpha).fit(train[M.FEATURES], train[M.TARGET])


def config_scores(degree, alpha=None):
    """Chỉ số trên tập kiểm thử của đúng cấu hình (bậc, alpha) lấy từ bảng đã tính."""
    if alpha is None:
        return poly_results().loc[degree]
    r = ridge_results()
    return r[(r["Degree"] == degree) & (r["alpha"] == alpha)].iloc[0]
