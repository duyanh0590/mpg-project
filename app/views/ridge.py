"""Trang: Ridge Regression – tác dụng của regularization (hệ số α) đối với Polynomial Regression."""
import plotly.graph_objects as go
import streamlit as st

import core
import modeling as M
import style


def render():
    style.header("Ridge Regression", "Regularization L2 kiểm soát overfitting của Polynomial Regression bậc cao")
    _, train, test = core.data()
    poly, ridge = core.poly_results(), core.ridge_results()
    best = M.best_ridge_per_degree(ridge)
    top = ridge.loc[ridge["CV RMSE"].idxmin()]

    a, b = st.columns([3, 2], vertical_alignment="center")
    a.markdown("Ridge Regression thêm vào hàm mất mát một **thành phần phạt** tỉ lệ với bình phương độ lớn các hệ số. "
               "α càng lớn, các hệ số càng bị thu nhỏ về 0: mô hình bớt linh hoạt, giảm variance nhưng tăng bias. "
               "Các đặc trưng được chuẩn hóa (StandardScaler) trước khi phạt để mọi hệ số được phạt công bằng.")
    b.latex(r"\min_{w}\; \sum_{i=1}^{n}\left(y_i - \hat y_i\right)^2 + \alpha \sum_{j=1}^{p} w_j^2")

    c = st.columns(4)
    style.kpi(c[0], "Ridge tốt nhất theo CV", f"Bậc {int(top['Degree'])}, α = {top['alpha']:.3g}",
              f"RMSE CV {top['CV RMSE']:.3f} · Test RMSE {top['Test RMSE']:.3f}", tone="navy text")
    style.kpi(c[1], "Bậc 10 không điều chuẩn", f"{poly.loc[10, 'Test RMSE']:.1f}", "Test RMSE (MPG)", tone="gray")
    style.kpi(c[2], f"Bậc 10 với Ridge (α = {best.loc[10, 'alpha']:.3g})", f"{best.loc[10, 'Test RMSE']:.2f}",
              f"Test RMSE, giảm {poly.loc[10, 'Test RMSE'] / best.loc[10, 'Test RMSE']:.0f} lần")
    style.kpi(c[3], "Giá trị α đã thử", f"{len(M.ALPHAS)}", "từ 10⁻⁴ đến 10⁴, thang log")
    st.write("")

    # ---- Ảnh hưởng của α ở một bậc ----
    st.subheader("Ảnh hưởng của α")
    d = st.select_slider("Bậc đa thức", options=M.DEGREES, value=10,
                         help="Bậc cao thấy rõ tác dụng của regularization nhất.")
    r = ridge[ridge["Degree"] == d]
    a_best = best.loc[d, "alpha"]
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_scatter(x=r["alpha"], y=r["Train RMSE"], name="Huấn luyện", mode="lines+markers", line_color=style.TRAIN)
        fig.add_scatter(x=r["alpha"], y=r["Test RMSE"], name="Kiểm thử", mode="lines+markers", line_color=style.TEST)
        fig.add_scatter(x=r["alpha"], y=r["CV RMSE"], name="CV 5-fold", mode="lines+markers", line=dict(color=style.CV, dash="dash"))
        # vẽ đường dọc bằng trace (shape/annotation của Plotly trên trục log dễ bị đặt sai vị trí)
        fig.add_scatter(x=[a_best, a_best], y=[0, max(8, r["Test RMSE"].max() * 1.05)], mode="lines",
                        name=f"α tốt nhất (CV) = {a_best:.3g}", line=dict(color=style.BLUE, dash="dot"))
        p_test = poly.loc[d, "Test RMSE"]
        y_max = max(8, r["Test RMSE"].max() * 1.05)
        if p_test < y_max:
            fig.add_hline(y=p_test, line_dash="dash", line_color=style.RED, annotation_text="Không điều chuẩn (kiểm thử)",
                          annotation_position="bottom left")
        fig.update_xaxes(title="α (thang log)", type="log", dtick=1, exponentformat="power")
        fig.update_yaxes(title="RMSE (MPG)", range=[0, y_max])
        style.chart(fig, title=f"Sai số theo α, bậc {d}")
        if p_test >= y_max:
            st.caption(f"Không điều chuẩn, bậc {d} có Test RMSE {p_test:.1f}, nằm ngoài trục.")
    with right:
        fig = go.Figure()
        fig.add_scatter(x=r["alpha"], y=r["||w||"], mode="lines+markers", name="||w||", line_color=style.NAVY)
        fig.update_xaxes(title="α (thang log)", type="log", dtick=1, exponentformat="power")
        fig.update_yaxes(title="Độ lớn vector hệ số ||w|| (thang log)", type="log", dtick=1, exponentformat="power")
        style.chart(fig, title="Hệ số bị thu nhỏ khi α tăng")
        st.caption("Ridge không đưa hệ số về đúng 0 (khác Lasso) mà thu nhỏ đều, nhất là các hệ số của đặc trưng tương quan cao.")

    lo, hi = r.iloc[0], r.iloc[-1]
    st.markdown(
        f"**Ba vùng của α (bậc {d}).**\n"
        f"- **α rất nhỏ ({lo['alpha']:.3g}):** gần như không điều chuẩn; Train RMSE {lo['Train RMSE']:.2f}, Test RMSE "
        f"{lo['Test RMSE']:.2f}, khoảng cách lớn là dấu hiệu overfitting.\n"
        f"- **α vừa phải (quanh {a_best:.3g}):** cân bằng bias–variance, RMSE CV thấp nhất {best.loc[d, 'CV RMSE']:.3f}, "
        f"Test RMSE {best.loc[d, 'Test RMSE']:.3f}.\n"
        f"- **α rất lớn ({hi['alpha']:.3g}):** phạt quá mạnh, hệ số bị ép gần về 0; Train RMSE {hi['Train RMSE']:.2f} và Test RMSE "
        f"{hi['Test RMSE']:.2f} cùng tăng, tức underfitting.")

    # ---- So sánh với Polynomial thường ở mọi bậc ----
    st.subheader("Polynomial Regression và Ridge Regression ở từng bậc")
    fig = go.Figure()
    fig.add_bar(x=poly.index, y=poly["Test RMSE"], name="Polynomial (không điều chuẩn)", marker_color=style.LIGHT_BLUE)
    fig.add_bar(x=best.index, y=best["Test RMSE"], name="Ridge (α tốt nhất theo CV)", marker_color=style.NAVY)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title="Bậc đa thức (degree)", dtick=1)
    fig.update_yaxes(title="Test RMSE (MPG, thang log)", type="log", tickvals=[1, 2, 3, 5, 10, 20, 50, 100, 200])
    style.chart(fig, title="Test RMSE theo bậc")

    tbl = best[["alpha", "Train R²", "Test R²", "Test MAE", "Test RMSE", "CV RMSE"]].copy()
    tbl.insert(0, "Polynomial Test RMSE", poly["Test RMSE"])
    st.dataframe(tbl.rename(columns={"alpha": "α tốt nhất", "Test MAE": "MAE", "Test RMSE": "Ridge Test RMSE"})
                 .style.format({"α tốt nhất": "{:.3g}"} | {c: "{:.3f}" for c in tbl.columns if c != "alpha"}),
                 width="stretch")
    span = best["CV RMSE"].max() - best["CV RMSE"].min()
    st.markdown(
        f"**Nhận xét.** Với α chọn bằng CV, mọi bậc từ 2 đến 10 đều cho RMSE CV trong khoảng "
        f"{best.loc[2:, 'CV RMSE'].min():.2f}–{best.loc[2:, 'CV RMSE'].max():.2f} (chênh {span:.3f}), trong khi Polynomial "
        f"không điều chuẩn tăng từ {poly.loc[2, 'CV RMSE']:.2f} lên {poly.loc[10, 'CV RMSE']:.1f}. Regularization gần như loại bỏ "
        f"overfitting. Bậc càng cao thì α tối ưu càng lớn (bậc 2: {best.loc[2, 'alpha']:.3g}, bậc 10: {best.loc[10, 'alpha']:.3g}) "
        "vì có nhiều đặc trưng tương quan hơn cần thu nhỏ. Chênh lệch giữa các bậc nhỏ hơn nhiều so với độ lệch chuẩn qua fold "
        f"(khoảng {best['CV RMSE std'].mean():.2f}), nên theo nguyên tắc chọn mô hình đơn giản, bậc thấp là đủ.")
