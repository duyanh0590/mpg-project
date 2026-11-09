"""Trang: Khám phá dữ liệu – bảng dữ liệu, giá trị thiếu, tương quan, quan hệ từng đặc trưng với MPG."""
import plotly.express as px
import streamlit as st

import core
import modeling as M
import style
from data import NUM


def render():
    style.header("Khám phá dữ liệu", "Bộ dữ liệu Auto MPG: thống kê mô tả, giá trị thiếu và quan hệ giữa các đặc trưng")
    df, _, _ = core.data()
    d = df.assign(**{"Xuất xứ": df["origin"].map(M.ORIGIN_VI)})

    c = st.columns(4)
    style.kpi(c[0], "Số xe", f"{len(df)}", f"{df['name'].nunique()} tên xe khác nhau")
    style.kpi(c[1], "Giá trị thiếu", f"{int(df.isna().sum().sum())}", "đều ở cột horsepower, điền bằng trung vị")
    style.kpi(c[2], "Năm sản xuất", f"19{df['model_year'].min()}–19{df['model_year'].max()}",
              "dữ liệu ghi dạng rút gọn 70–82")
    style.kpi(c[3], "Xuất xứ", " · ".join(f"{M.ORIGIN_VI[k]} {v}" for k, v in df["origin"].value_counts().items()),
              "số xe theo nước", tone="gray text")
    st.write("")

    t1, t2, t3, t4 = st.tabs(["Bảng dữ liệu", "Tương quan", "Quan hệ với MPG", "MPG theo năm"])
    with t1:
        chon = st.multiselect("Lọc theo xuất xứ", list(M.ORIGIN_VI.values()), default=list(M.ORIGIN_VI.values()))
        st.dataframe(d[d["Xuất xứ"].isin(chon)].drop(columns="origin"), width="stretch", height=320, hide_index=True)
        st.markdown("**Thống kê mô tả**")
        st.dataframe(df[["mpg"] + NUM].describe().T.round(2), width="stretch")
    with t2:
        corr = df[["mpg"] + NUM].corr().round(2)
        fig = px.imshow(corr, text_auto=True, zmin=-1, zmax=1, color_continuous_scale=["#c0392b", "#ffffff", style.BLUE],
                        aspect="auto")
        style.chart(fig, height=460, title="Hệ số tương quan Pearson")
        st.caption("MPG tương quan âm mạnh với weight, displacement, horsepower, cylinders và dương với model_year. "
                   "Bốn đặc trưng cylinders, displacement, horsepower, weight tương quan rất cao với nhau (đa cộng tuyến), "
                   "nên nhánh đa thức chỉ dùng weight, horsepower, acceleration, model_year.")
    with t3:
        x = st.selectbox("Đặc trưng", NUM, index=3, format_func=M.LABELS.get)
        a, b = st.columns([3, 2])
        with a:
            fig = px.scatter(d, x=x, y="mpg", color="Xuất xứ", hover_name="name", color_discrete_map=style.ORIGIN_COLORS,
                             labels=M.LABELS, opacity=.75)
            style.chart(fig, title=f"MPG theo {M.LABELS[x]}")
        with b:
            fig = px.box(d, x="Xuất xứ", y="mpg", color="Xuất xứ", color_discrete_map=style.ORIGIN_COLORS, labels=M.LABELS)
            fig.update_layout(showlegend=False)
            style.chart(fig, title="MPG theo xuất xứ")
    with t4:
        g = d.groupby(["model_year", "Xuất xứ"], as_index=False)["mpg"].mean()
        fig = px.line(g, x="model_year", y="mpg", color="Xuất xứ", markers=True, color_discrete_map=style.ORIGIN_COLORS,
                      labels={"model_year": "Model Year (70 = 1970)", "mpg": "MPG trung bình"})
        style.chart(fig, title="MPG trung bình theo năm sản xuất")
        st.caption("Xe đời sau tiết kiệm nhiên liệu hơn rõ rệt, nhất là sau khủng hoảng dầu mỏ 1973 và 1979.")
