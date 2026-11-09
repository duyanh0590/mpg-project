"""Trang 1: Tổng quan – các chỉ số chính, phân bố MPG, quan hệ MPG – trọng lượng."""
import plotly.express as px
import streamlit as st

import core
import modeling as M
import style


def render():
    style.header("AutoMPG Analytics", "Phân tích và dự đoán hiệu suất nhiên liệu ô tô bằng Machine Learning")
    df, train, test = core.data()
    cand = M.candidates(core.poly_results(), core.ridge_results())

    st.markdown(
        "**Mục tiêu.** Dự đoán mức tiêu hao nhiên liệu (MPG, dặm/gallon) của ô tô từ thông số kỹ thuật, và nghiên cứu "
        "hai hiện tượng cốt lõi của học máy: **underfitting/overfitting** khi tăng bậc của Polynomial Regression, "
        "và tác dụng của **regularization** (Ridge Regression) trong việc kiểm soát overfitting.")
    style.steps([
        ("Dữ liệu", "Auto MPG, 398 xe, 6 giá trị horsepower bị thiếu"),
        ("Tiền xử lý", "Điền trung vị, PolynomialFeatures, StandardScaler, one-hot origin"),
        ("Huấn luyện", "Polynomial bậc 1–10 và Ridge với 17 giá trị α"),
        ("Đánh giá", "MAE, RMSE, R² trên tập kiểm thử và CV 5-fold"),
        ("Dự đoán", "Nhập thông số xe, chọn mô hình, xem kết quả"),
    ])
    st.write("")

    # ---- Chỉ số dữ liệu ----
    c = st.columns(4)
    style.kpi(c[0], "Tổng số mẫu", f"{len(df)}", f"{len(train)} huấn luyện · {len(test)} kiểm thử")
    style.kpi(c[1], "Đặc trưng đầu vào", "7", "6 thuộc tính số + origin")
    style.kpi(c[2], "MPG trung bình", f"{df['mpg'].mean():.2f}", f"Thấp nhất {df['mpg'].min():g} · cao nhất {df['mpg'].max():g}")
    style.kpi(c[3], "Số mô hình đã huấn luyện", f"{len(core.poly_results()) + len(core.ridge_results())}",
              "10 Polynomial + 170 Ridge (10 bậc × 17 α)", tone="gray")

    # ---- Mô hình tốt nhất ----
    st.subheader("Mô hình tốt nhất trên tập kiểm thử")
    left, right = st.columns([1, 3], vertical_alignment="center")
    metric = left.selectbox("Chỉ số dùng để chọn", ["RMSE", "MAE", "R²"],
                            help="Ứng viên: Polynomial bậc 1–10 và Ridge bậc 1–10 (mỗi bậc lấy α tốt nhất theo CV).")
    col = f"Test {metric}"
    best = cand.loc[cand[col].idxmax() if metric == "R²" else cand[col].idxmin()]
    c = right.columns([1.6, 1, 1, 1])
    style.kpi(c[0], "Mô hình", best["Model"], f"tốt nhất theo Test {metric}", tone="navy text")
    style.kpi(c[1], "MAE", f"{best['Test MAE']:.3f}", "MPG, trên tập kiểm thử")
    style.kpi(c[2], "RMSE", f"{best['Test RMSE']:.3f}", "MPG, trên tập kiểm thử")
    style.kpi(c[3], "R²", f"{best['Test R²']:.3f}", "trên tập kiểm thử")

    cv_best = cand.loc[cand["CV RMSE"].idxmin()]
    st.caption(
        f"Lưu ý phương pháp: tập kiểm thử chỉ có {len(test)} xe nên xếp hạng theo nó dao động nhiều. Báo cáo chọn mô hình theo "
        f"RMSE CV 5-fold trên tập huấn luyện; theo cách đó mô hình tốt nhất là **{cv_best['Model']}** "
        f"(RMSE CV {cv_best['CV RMSE']:.3f}, Test RMSE {cv_best['Test RMSE']:.3f}). "
        "Chương 3 của báo cáo cải tiến tiếp bằng đặc trưng mới và Voting (Ridge + Gradient Boosting).")

    # ---- Biểu đồ ----
    d = df.assign(**{"Xuất xứ": df["origin"].map(M.ORIGIN_VI)})
    a, b = st.columns(2)
    with a:
        fig = px.histogram(d, x="mpg", nbins=30, color_discrete_sequence=[style.BLUE], labels={"mpg": "MPG"})
        fig.add_vline(x=d["mpg"].mean(), line_dash="dash", line_color=style.NAVY,
                      annotation_text=f"Trung bình {d['mpg'].mean():.1f}", annotation_position="top right")
        fig.update_traces(marker_line_color="white", marker_line_width=1)
        fig.update_yaxes(title="Số xe")
        style.chart(fig, title="Phân bố MPG")
    with b:
        fig = px.scatter(d, x="weight", y="mpg", color="Xuất xứ", hover_name="name", color_discrete_map=style.ORIGIN_COLORS,
                         hover_data={"model_year": True, "horsepower": True}, labels=M.LABELS, opacity=.75)
        style.chart(fig, title="MPG theo trọng lượng xe")
    st.caption(f"MPG lệch phải (đa số xe 15–30 MPG). Xe càng nặng càng tốn nhiên liệu: hệ số tương quan MPG – weight là "
               f"{df['mpg'].corr(df['weight']):.2f}, và quan hệ có dạng cong, là lý do thử Polynomial Regression.")
