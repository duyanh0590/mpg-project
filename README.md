# Dự đoán mức tiêu hao nhiên liệu của ô tô (Auto MPG) – Dự án học phần Học máy

Bài toán hồi quy: dự đoán `mpg` (dặm/gallon) từ thông số kỹ thuật của xe.
Nội dung: hồi quy đa thức bậc 1–10, underfitting/overfitting, bias–variance, Ridge, KNN.

## Cấu trúc
```
data/raw/        dữ liệu gốc (không sửa), kèm SOURCE.md
notebooks/       01_eda, 02_preprocessing, ... (chạy theo thứ tự số)
src/             data.py (đọc, chia tập), preprocess.py (Pipeline), baseline_reference.py
reports/         hình (figures/), bảng kết quả (.csv), nhật ký đóng góp
```

## Cách chạy
```
python -m venv .venv
.venv\Scripts\activate           # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
cd notebooks && jupyter lab      # mở từng notebook, chọn Kernel > Restart & Run All
python src/baseline_reference.py # chạy từ thư mục gốc dự án
```
- Mọi thí nghiệm dùng `random_state=42` (hằng `SEED` trong `src/data.py`).
- Dùng đường dẫn tương đối; notebook phải được mở từ thư mục `notebooks/`.
- Môi trường đã kiểm tra: Python 3.13, pandas 3.0, scikit-learn 1.9. Bản pandas 2.x vẫn chạy, chỉ khác cách in kiểu dữ liệu chuỗi (`object` thay vì `str`).

## Quy ước chia tập
Chia ~80/20, phân tầng theo `origin`, gom theo `name` (cùng tên xe không nằm ở cả hai tập). Tập kiểm tra chỉ dùng để báo cáo cuối. Chi tiết ở `notebooks/02_preprocessing.ipynb`.

## Tiến độ
Xem `reports/nhat_ky_dong_gop.md` và file kế hoạch 9 tuần.
