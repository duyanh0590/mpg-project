"""Đọc dữ liệu Auto MPG và chia tập huấn luyện/kiểm tra. Mọi notebook dùng chung để có cùng một cách chia."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
NUM = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year"]
CAT = ["origin"]
TARGET = "mpg"


def load_raw() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "raw" / "auto-mpg.csv")


def split(df: pd.DataFrame | None = None):
    """Chia ~80/20: phân tầng theo origin, gom theo name (cùng tên xe không nằm ở cả hai tập).

    Dùng fold đầu của StratifiedGroupKFold(5) nên tập kiểm tra xấp xỉ 20%, không đúng tuyệt đối 80 mẫu.
    """
    df = load_raw() if df is None else df
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(sgkf.split(df, df["origin"], groups=df["name"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
