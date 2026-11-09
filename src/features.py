"""Đặc trưng bổ sung cho bước cải tiến (notebook 05). Mỗi đặc trưng chỉ dùng thông tin của chính dòng đó nên không rò rỉ."""
import numpy as np
import pandas as pd

LOG = ["log_weight", "log_horsepower", "log_displacement"]
RATIO = ["hp_per_ton"]
DIESEL = ["diesel"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["weight", "horsepower", "displacement"]:
        df[f"log_{c}"] = np.log(df[c])                       # quan hệ với MPG gần dạng lũy thừa (mục 2.2.3)
    df["hp_per_ton"] = df["horsepower"] / df["weight"] * 2000  # công suất trên mỗi tấn Mỹ; thiếu horsepower -> NaN, Imputer xử lý
    df["diesel"] = df["name"].str.contains("diesel").astype(int)  # loại nhiên liệu là thông số kỹ thuật, chỉ ghi trong tên xe
    return df
