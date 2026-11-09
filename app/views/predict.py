"""Trang: Dự đoán MPG – nhập thông số xe, chọn Polynomial hoặc Ridge, xem kết quả từ mô hình đã huấn luyện."""
import numpy as np
import pandas as pd
import streamlit as st

import core
import modeling as M
import style
from data import NUM

L100 = 235.215          # L/100 km = 235.215 / MPG


def check_inputs(df, train, x):
    """Kiểm tra dữ liệu nhập. Trả về danh sách cảnh báo (chuỗi)."""
    warn = []
    for f in M.POLY:
        lo, hi = train[f].min(), train[f].max()
        if not lo <= x[f] <= hi:
            warn.append(f"{M.LABELS[f]} = {x[f]:g} nằm ngoài miền dữ liệu huấn luyện ({lo:g}–{hi:g}): mô hình phải ngoại suy.")
    same = df[df["cylinders"] == x["cylinders"]]
    if len(same) < 10:
        warn.append(f"Dữ liệu chỉ có {len(same)} xe {x['cylinders']} xi-lanh, kết quả so sánh nhóm kém tin cậy.")
    if len(same) and not same["displacement"].min() <= x["displacement"] <= same["displacement"].max():
        warn.append(f"Xe {x['cylinders']} xi-lanh trong dữ liệu có displacement {same['displacement'].min():g}–"
                    f"{same['displacement'].max():g} in³; giá trị {x['displacement']:g} là tổ hợp không thực tế.")
    if len(same) and x["horsepower"] > same["horsepower"].max() * 1.1:
        warn.append(f"Công suất {x['horsepower']:g} hp cao bất thường cho xe {x['cylinders']} xi-lanh "
                    f"(tối đa trong dữ liệu {same['horsepower'].max():g} hp).")
    return warn


def model_config():
    """Khu vực chọn phương pháp, bậc đa thức và α. Trả về (tên, bậc, alpha)."""
    with st.container(border=True):
        st.markdown("**Cấu hình mô hình**")
        a, b, c = st.columns([1.3, 1, 1])
        method = a.radio("Phương pháp", ["Polynomial Regression", "Ridge Regression"], horizontal=True)
        if method == "Polynomial Regression":
            degree = b.slider("Bậc đa thức (degree)", 1, 10, 2, key="deg_poly")
            alpha = None
            c.caption("Polynomial Regression không có tham số điều chuẩn. Bậc 2 cho RMSE CV thấp nhất; "
                      "bậc ≥ 5 bị overfitting nặng.")
        else:
            degree = b.slider("Bậc đa thức (degree)", 1, 10, 6, key="deg_ridge")
            opts = list(M.ALPHAS)
            alpha = c.select_slider("Hệ số điều chuẩn α", options=opts, value=opts[10], format_func=lambda v: f"{v:.3g}")
    return method, degree, alpha


def render():
    style.header("Dự đoán MPG", "Nhập thông số kỹ thuật của xe, chọn mô hình và nhấn Dự đoán")
    df, train, test = core.data()
    med = train[NUM].median()

    method, degree, alpha = model_config()

    with st.form("input"):
        st.markdown("**Thông số xe**")
        c1, c2, c3 = st.columns(3)
        cyl_opts = sorted(df["cylinders"].unique().tolist())
        x = {
            "cylinders": c1.selectbox("Cylinders (số xi-lanh)", cyl_opts, index=cyl_opts.index(4)),
            "displacement": c1.number_input("Displacement – dung tích động cơ (in³)", 50.0, 500.0,
                                            float(med["displacement"]), 5.0),
            "origin": c1.selectbox("Origin – xuất xứ", list(M.ORIGIN_VI), format_func=M.ORIGIN_VI.get),
            "horsepower": c2.number_input("Horsepower – công suất (hp)", 40.0, 250.0, float(med["horsepower"]), 5.0),
            "weight": c2.number_input("Weight – trọng lượng (lb)", 1500, 5500, int(med["weight"]), 50),
            "acceleration": c3.number_input("Acceleration – thời gian tăng tốc 0–60 mph (giây)", 8.0, 25.0,
                                            float(med["acceleration"]), 0.5),
            "model_year": c3.slider("Model Year – năm sản xuất (70 = 1970, 82 = 1982)", 70, 82, int(med["model_year"])),
        }
        st.caption("Giá trị mặc định là trung vị của tập huấn luyện. Mô hình dùng weight, horsepower, acceleration, "
                   "model_year và origin; cylinders và displacement dùng để kiểm tra tính hợp lý và so sánh với xe tương tự.")
        submitted = st.form_submit_button("Dự đoán", type="primary")

    if submitted:
        model = core.fitted_model(degree, alpha)
        pred = float(model.predict(M.car(**x))[0])
        st.session_state.result = dict(x=x, method=method, degree=degree, alpha=alpha, pred=pred)

    r = st.session_state.get("result")
    if r is None:
        st.info("Nhập thông số và nhấn **Dự đoán** để xem kết quả.")
        return
    show_result(df, train, r)


def show_result(df, train, r):
    x, pred, degree, alpha = r["x"], r["pred"], r["degree"], r["alpha"]
    name = f"{r['method']}, bậc {degree}" + (f", α = {alpha:.3g}" if alpha is not None else "")
    sc = core.config_scores(degree, alpha)
    mean_mpg = df["mpg"].mean()
    valid = 0 < pred < 80

    st.subheader("Kết quả")
    a, b = st.columns([1.3, 2])
    with a:
        if valid:
            st.markdown(f'<div class="result"><div class="label">MPG dự đoán</div>'
                        f'<div class="value">{pred:.1f} <small>MPG</small></div>'
                        f'<div class="note">≈ {L100 / pred:.1f} L/100 km · {name}</div></div>', unsafe_allow_html=True)
        else:
            st.error(f"Mô hình {name} cho kết quả {pred:.1f} MPG, không có ý nghĩa vật lý. Đây là biểu hiện của overfitting "
                     "hoặc ngoại suy ngoài miền dữ liệu; hãy giảm bậc đa thức hoặc dùng Ridge Regression.")
    with b:
        c = st.columns(3)
        if valid:
            style.kpi(c[0], "So với MPG trung bình", f"{pred - mean_mpg:+.1f}",
                      f"trung bình bộ dữ liệu {mean_mpg:.1f} MPG ({(pred / mean_mpg - 1) * 100:+.0f}%)")
        grp = df[(df["cylinders"] == x["cylinders"]) & (df["origin"] == x["origin"])]
        style.kpi(c[1], "Xe cùng nhóm", f"{grp['mpg'].mean():.1f}" if len(grp) else "–",
                  f"MPG TB của {len(grp)} xe {x['cylinders']} xi-lanh, {M.ORIGIN_VI[x['origin']]}", tone="gray")
        style.kpi(c[2], "Sai số điển hình", f"±{sc['Test RMSE']:.2f}",
                  f"RMSE của cấu hình này trên tập kiểm thử (R² {sc['Test R²']:.3f})", tone="navy")

    for w in check_inputs(df, train, x):
        st.warning(w)
    if sc["Test R²"] < 0.5:
        st.warning(f"Cấu hình {name} có Test R² = {sc['Test R²']:.2f}: mô hình overfitting nặng, dự đoán không đáng tin "
                   "dù con số trông hợp lý.")
    if valid:
        lo, hi = max(pred - sc["Test RMSE"], 0.1), pred + sc["Test RMSE"]
        st.markdown(f"**Giải thích.** Với thông số đã nhập, mô hình dự đoán xe đi được khoảng **{pred:.1f} dặm cho mỗi gallon** "
                    f"nhiên liệu ({L100 / pred:.1f} lít cho 100 km), "
                    f"{'tiết kiệm hơn' if pred > mean_mpg else 'tốn nhiên liệu hơn'} mức trung bình của bộ dữ liệu. "
                    f"Sai số điển hình của cấu hình này khoảng ±{sc['Test RMSE']:.1f} MPG, nên giá trị thực tế thường nằm "
                    f"trong khoảng {lo:.1f}–{hi:.1f} MPG.")

    a, b = st.columns(2)
    with a:
        st.markdown("**Thông số đầu vào đã sử dụng**")
        tbl = pd.DataFrame({
            "Thông số": [M.LABELS[k] for k in x],
            "Giá trị": [M.ORIGIN_VI[v] if k == "origin" else (f"19{v}" if k == "model_year" else f"{v:g}") for k, v in x.items()],
            "Dùng trong mô hình": ["Có" if k in M.FEATURES else "Không (chỉ kiểm tra)" for k in x],
        })
        st.dataframe(tbl, hide_index=True, width="stretch")
    with b:
        st.markdown("**5 xe thật gần nhất trong dữ liệu** (theo weight, horsepower, acceleration, model_year)")
        z = (df[M.POLY] - train[M.POLY].mean()) / train[M.POLY].std()
        q = (pd.Series({f: x[f] for f in M.POLY}) - train[M.POLY].mean()) / train[M.POLY].std()
        near = df.loc[np.sqrt(((z - q) ** 2).sum(axis=1)).nsmallest(5).index]
        st.dataframe(near[["name", "model_year", "weight", "horsepower", "origin", "mpg"]]
                     .assign(origin=lambda t: t["origin"].map(M.ORIGIN_VI))
                     .rename(columns={"name": "Tên xe", "model_year": "Năm", "weight": "Weight", "horsepower": "HP",
                                      "origin": "Xuất xứ", "mpg": "MPG thực tế"}), hide_index=True, width="stretch")

    with st.expander("Giới hạn của dự đoán"):
        st.markdown(
            "- Dữ liệu chỉ gồm 398 xe đời 1970–1982 (chủ yếu thị trường Mỹ); xe hiện đại, xe hybrid, xe điện nằm ngoài phạm vi.\n"
            "- MPG theo đơn vị gallon Mỹ; quy đổi L/100 km = 235.215 / MPG.\n"
            "- Mô hình học mối tương quan, không phải quan hệ nhân quả: thay một thông số trong khi giữ nguyên các thông số "
            "khác có thể tạo ra một chiếc xe không tồn tại.\n"
            "- Ngoài miền dữ liệu huấn luyện, đa thức bậc cao ngoại suy rất kém; Ridge giảm bớt nhưng không loại bỏ hoàn toàn.\n"
            "- Sai số điển hình lấy từ 80 xe của tập kiểm thử nên bản thân nó cũng có sai số.")
