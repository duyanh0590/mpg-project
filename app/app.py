"""AutoMPG Analytics – ứng dụng Streamlit dự đoán MPG bằng Polynomial Regression và Ridge Regression.

Chạy từ thư mục gốc dự án (mpg-project), đã kích hoạt môi trường ảo:
    streamlit run app/app.py
Cấu trúc:
    app.py        điều hướng giữa các trang (file này)
    modeling.py   huấn luyện, đánh giá mô hình (không phụ thuộc Streamlit)
    core.py       lưu cache dữ liệu và kết quả huấn luyện
    style.py      màu sắc, thẻ KPI, định dạng biểu đồ
    views/        mỗi trang một file
"""
import streamlit as st

import core
import style
from views import explore, overview, polynomial, predict, ridge

st.set_page_config(page_title="AutoMPG Analytics", page_icon=":material/local_gas_station:", layout="wide")
style.setup()

pages = [
    st.Page(overview.render, title="Tổng quan", icon=":material/dashboard:", url_path="tong-quan", default=True),
    st.Page(explore.render, title="Khám phá dữ liệu", icon=":material/table_chart:", url_path="du-lieu"),
    st.Page(predict.render, title="Dự đoán MPG", icon=":material/speed:", url_path="du-doan"),
    st.Page(polynomial.render, title="Polynomial Regression", icon=":material/show_chart:", url_path="polynomial"),
    st.Page(ridge.render, title="Ridge Regression", icon=":material/tune:", url_path="ridge"),
]

nav = st.navigation(pages, position="hidden")   # tự vẽ menu bên dưới để tên ứng dụng nằm trên cùng

with st.sidebar:
    st.markdown("### AutoMPG Analytics")
    st.caption("Phân tích và dự đoán hiệu suất nhiên liệu ô tô bằng Machine Learning")
    st.divider()
    for p in pages:
        st.page_link(p)
    st.divider()
    df, train, test = core.data()
    st.caption(f"Dữ liệu: Auto MPG (UCI), {len(df)} xe đời 1970–1982.  \n"
               f"Chia tập: {len(train)} huấn luyện / {len(test)} kiểm thử, phân tầng theo origin, "
               "cùng tên xe không nằm ở cả hai tập.  \n"
               "Đồ án môn Học máy – Đào Duy Anh, Lý A Cường.")

nav.run()
