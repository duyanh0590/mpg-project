"""Màu sắc, CSS và các hàm vẽ giao diện dùng chung (thẻ KPI, tiêu đề trang, định dạng biểu đồ Plotly)."""
import streamlit as st

NAVY = "#0b2545"
BLUE = "#1f6fd1"
LIGHT_BLUE = "#8db8f2"
GRAY = "#8a94a6"
RED = "#c0392b"
# Màu cố định cho từng đường để mọi biểu đồ thống nhất
TRAIN, TEST, CV = LIGHT_BLUE, NAVY, GRAY
ORIGIN_COLORS = {"Mỹ": NAVY, "Châu Âu": BLUE, "Nhật Bản": LIGHT_BLUE}

CSS = f"""
<style>
.block-container {{ padding-top: 2rem; max-width: 1280px; }}
h1, h2, h3 {{ color: {NAVY}; }}
.app-header {{ border-bottom: 1px solid #dde3ec; padding-bottom: .6rem; margin-bottom: 1.2rem; }}
.app-header .title {{ font-size: 1.75rem; font-weight: 700; color: {NAVY}; margin: 0; }}
.app-header .sub {{ color: #5b6779; margin: .15rem 0 0 0; }}
.kpi {{ background: #fff; border: 1px solid #dde3ec; border-left: 4px solid {BLUE}; border-radius: 10px;
        padding: .8rem 1rem; height: 100%; }}
.kpi .label {{ color: #5b6779; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }}
.kpi .value {{ color: {NAVY}; font-size: 1.55rem; font-weight: 700; line-height: 1.3; }}
.kpi .note {{ color: #7a8699; font-size: .8rem; }}
.kpi.navy {{ border-left-color: {NAVY}; }}
.kpi.gray {{ border-left-color: {GRAY}; }}
.kpi.text .value {{ font-size: 1.15rem; padding: .25rem 0; }}
.result {{ background: {NAVY}; color: #fff; border-radius: 12px; padding: 1.2rem 1.4rem; }}
.result .label {{ color: #b9cbe6; font-size: .85rem; text-transform: uppercase; letter-spacing: .05em; }}
.result .value {{ font-size: 2.6rem; font-weight: 700; line-height: 1.2; }}
.result .value small {{ font-size: 1.1rem; font-weight: 400; color: #b9cbe6; }}
.result .note {{ color: #d5e1f2; font-size: .9rem; }}
.step {{ background: #fff; border: 1px solid #dde3ec; border-radius: 10px; padding: .7rem .9rem; height: 100%; }}
.step .n {{ color: {BLUE}; font-weight: 700; font-size: .8rem; }}
.step .t {{ color: {NAVY}; font-weight: 600; }}
.step .d {{ color: #5b6779; font-size: .85rem; }}
</style>
"""


def setup():
    st.markdown(CSS, unsafe_allow_html=True)


def header(title, subtitle):
    st.markdown(f'<div class="app-header"><p class="title">{title}</p><p class="sub">{subtitle}</p></div>',
                unsafe_allow_html=True)


def kpi(col, label, value, note="", tone=""):
    col.markdown(f'<div class="kpi {tone}"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="note">{note}</div></div>', unsafe_allow_html=True)


def steps(items):
    """Các bước quy trình dạng thẻ nằm ngang. items: [(tiêu đề, mô tả), ...]"""
    for i, (col, (t, d)) in enumerate(zip(st.columns(len(items)), items), start=1):
        col.markdown(f'<div class="step"><div class="n">BƯỚC {i}</div><div class="t">{t}</div><div class="d">{d}</div></div>',
                     unsafe_allow_html=True)


def chart(fig, height=380, title=None):
    """Định dạng chung cho mọi biểu đồ Plotly rồi hiển thị."""
    fig.update_layout(template="plotly_white", height=height,
                      title=dict(text=title, x=0, y=.98, yanchor="top", font=dict(color=NAVY, size=16)),
                      font=dict(family="sans-serif", size=13, color="#1b2a41"),
                      margin=dict(l=10, r=10, t=80 if title else 30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title=None),
                      paper_bgcolor="#ffffff", plot_bgcolor="#ffffff")
    fig.update_xaxes(gridcolor="#eef1f6", zeroline=False)
    fig.update_yaxes(gridcolor="#eef1f6", zeroline=False)
    with st.container(border=True):
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
