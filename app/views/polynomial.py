"""Trang: Polynomial Regression bậc 1–10 – underfitting, overfitting và bias–variance."""
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import core
import modeling as M
import style


def render():
    style.header("So sánh Polynomial Regression",
                 "Huấn luyện bậc 1–10 trên cùng tập dữ liệu để thấy underfitting và overfitting")
    df, train, test = core.data()
    poly = core.poly_results()
    d_cv, d_test = int(poly["CV RMSE"].idxmin()), int(poly["Test RMSE"].idxmin())

    st.markdown(
        "Polynomial Regression bậc *d* sinh mọi tích của các đặc trưng có tổng bậc ≤ *d* (PolynomialFeatures) rồi khớp "
        "hồi quy tuyến tính bằng bình phương tối thiểu. Đặc trưng sinh đa thức: weight, horsepower, acceleration, "
        "model_year; origin được one-hot và không nhân đa thức. Bậc càng cao, mô hình càng linh hoạt nhưng càng dễ học cả nhiễu.")

    c = st.columns(4)
    style.kpi(c[0], "Bậc tốt nhất theo CV", f"{d_cv}", f"RMSE CV {poly.loc[d_cv, 'CV RMSE']:.3f} (cách chọn trong báo cáo)",
              tone="navy")
    style.kpi(c[1], "Bậc tốt nhất theo Test RMSE", f"{d_test}", f"Test RMSE {poly.loc[d_test, 'Test RMSE']:.3f}")
    style.kpi(c[2], "Bậc 1 (underfitting)", f"{poly.loc[1, 'Test RMSE']:.2f}",
              f"Test RMSE · Train RMSE {poly.loc[1, 'Train RMSE']:.2f}, cùng cao", tone="gray")
    style.kpi(c[3], "Bậc 10 (overfitting)", f"{poly.loc[10, 'Test RMSE']:.1f}",
              f"Test RMSE · {poly.loc[10, 'Số đặc trưng']} đặc trưng cho {len(train)} mẫu", tone="gray")
    st.write("")

    # ---- Bảng kết quả ----
    st.subheader("Kết quả từng bậc")
    cols = ["Số đặc trưng", "Train R²", "Test R²", "Test MAE", "Test RMSE", "Train RMSE", "CV RMSE"]
    tbl = poly[cols].rename(columns={"Test MAE": "MAE", "Test RMSE": "RMSE"})
    styled = (tbl.style.format({c: "{:.3f}" for c in tbl.columns if c != "Số đặc trưng"})
              .apply(lambda r: [f"background-color: #e3eefc; font-weight: 600" if r.name == d_cv else "" for _ in r], axis=1))
    st.dataframe(styled, width="stretch")
    st.caption(f"MAE, RMSE (đơn vị MPG) và Test R² tính trên {len(test)} xe của tập kiểm thử; CV RMSE là trung bình 5 fold "
               f"trên tập huấn luyện. Dòng tô đậm: bậc {d_cv}, tốt nhất theo CV.")

    # ---- Biểu đồ sai số ----
    log = st.toggle("Trục RMSE dạng log", value=True, help="Bậc cao có RMSE lớn gấp hàng chục lần, cần trục log để thấy rõ bậc thấp.")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        fig.add_scatter(x=poly.index, y=poly["Train RMSE"], name="Huấn luyện", mode="lines+markers", line_color=style.TRAIN)
        fig.add_scatter(x=poly.index, y=poly["Test RMSE"], name="Kiểm thử", mode="lines+markers", line_color=style.TEST)
        fig.add_scatter(x=poly.index, y=poly["CV RMSE"], name="CV 5-fold", mode="lines+markers",
                        line=dict(color=style.CV, dash="dash"))
        if d_cv > 1:
            fig.add_vrect(x0=.5, x1=d_cv - .5, fillcolor=style.GRAY, opacity=.08, line_width=0,
                          annotation_text="Underfitting", annotation_position="top left")
        fig.add_vrect(x0=d_cv + .5, x1=10.5, fillcolor=style.RED, opacity=.05, line_width=0,
                      annotation_text="Overfitting", annotation_position="top right")
        fig.update_xaxes(title="Bậc đa thức (degree)", dtick=1)
        if log:
            fig.update_yaxes(type="log", tickvals=[1, 2, 3, 5, 10, 20, 50, 100, 200])
        fig.update_yaxes(title="RMSE (MPG)")
        style.chart(fig, title="Sai số huấn luyện và kiểm thử theo bậc")
    with b:
        fig = go.Figure()
        fig.add_scatter(x=poly.index, y=poly["Train R²"], name="Train R²", mode="lines+markers", line_color=style.TRAIN)
        fig.add_scatter(x=poly.index, y=poly["Test R²"], name="Test R²", mode="lines+markers", line_color=style.TEST)
        fig.update_xaxes(title="Bậc đa thức (degree)", dtick=1)
        fig.update_yaxes(title="R²", range=[-1.05, 1.05])
        style.chart(fig, title="Hiệu suất (R²) theo bậc")
        out = poly.index[poly["Test R²"] < -1].tolist()
        if out:
            st.caption(f"Trục R² giới hạn từ −1 đến 1; Test R² ở bậc {', '.join(map(str, out))} còn thấp hơn −1 "
                       f"(bậc 10: {poly.loc[10, 'Test R²']:.0f}), xem bảng.")

    gap = poly["CV RMSE"] - poly["Train RMSE"]       # dùng CV (5 fold) thay vì 80 xe kiểm thử để ổn định hơn
    st.markdown(
        f"**Nhận xét.**\n"
        f"- Sai số huấn luyện giảm đều từ {poly.loc[1, 'Train RMSE']:.2f} (bậc 1) xuống {poly.loc[10, 'Train RMSE']:.2f} (bậc 10): "
        f"mô hình nhiều tham số hơn luôn khớp tập huấn luyện tốt hơn.\n"
        f"- **Underfitting ở bậc 1:** sai số huấn luyện ({poly.loc[1, 'Train RMSE']:.2f}) và kiểm thử ({poly.loc[1, 'Test RMSE']:.2f}) "
        f"cùng cao và gần nhau, mô hình tuyến tính bỏ sót độ cong của dữ liệu (bias cao).\n"
        f"- **Cân bằng tốt nhất ở bậc {d_cv}–{max(d_cv, d_test)}:** RMSE CV thấp nhất ở bậc {d_cv}; trên tập kiểm thử bậc {d_test} "
        f"thấp nhất. Hai cách chọn khác nhau một bậc vì tập kiểm thử chỉ có {len(test)} xe.\n"
        f"- **Overfitting từ bậc {d_cv + 1}:** khoảng cách giữa RMSE CV và RMSE huấn luyện tăng từ {gap[d_cv + 1]:.2f} lên {gap[10]:.1f} ở bậc 10 "
        f"(variance cao). Từ bậc 8, số đặc trưng ({poly.loc[8, 'Số đặc trưng']}) vượt số mẫu huấn luyện ({len(train)}), "
        f"mô hình khớp cả nhiễu và ngoại suy rất kém.")

    # ---- Minh họa một chiều ----
    st.subheader("Minh họa trực quan: MPG theo weight")
    st.caption("Khớp đa thức chỉ với một đặc trưng weight để nhìn thấy hình dạng đường hồi quy khi đổi bậc.")
    deg = st.slider("Bậc đa thức cho hình minh họa", 1, 10, 2, key="demo_deg")
    w, p = M.weight_curve(train, deg)
    fig = px.scatter(train, x="weight", y="mpg", opacity=.45, color_discrete_sequence=[style.LIGHT_BLUE], labels=M.LABELS)
    fig.add_scatter(x=w, y=p, mode="lines", name=f"Đa thức bậc {deg}", line=dict(color=style.NAVY, width=3))
    fig.update_yaxes(range=[train["mpg"].min() - 5, train["mpg"].max() + 5])
    style.chart(fig, height=360)
    st.caption("Bậc 1 là đường thẳng, không theo kịp độ cong (underfitting). Bậc 2–3 bám theo xu hướng chung. "
               "Bậc cao bắt đầu uốn lượn ở hai đầu, nơi có ít xe: dấu hiệu đầu tiên của overfitting.")
