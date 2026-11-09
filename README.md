# Dự đoán mức tiêu hao nhiên liệu của ô tô (Auto MPG) – Dự án học phần Học máy

Bài toán hồi quy: dự đoán `mpg` (dặm/gallon) từ thông số kỹ thuật của xe.
Nội dung: hồi quy đa thức bậc 1–10, underfitting/overfitting, bias–variance, Ridge, KNN; cải tiến bằng đặc trưng mới, Random Forest, Gradient Boosting, Voting.

## Cấu trúc
```
data/raw/        dữ liệu gốc (không sửa), kèm SOURCE.md
notebooks/       chạy theo thứ tự số:
                 01_eda              khảo sát dữ liệu, vấn đề chất lượng
                 02_preprocessing    chia tập, xử lý, kiểm tra rò rỉ
                 03_models           baseline, đa thức bậc 1–10, Ridge, KNN, phần dư
                 04_analysis         learning curve, bias–variance bootstrap, phân tích lỗi, mức ảnh hưởng đặc trưng
                 05_improvement      ablation đặc trưng, RF/GB/Voting, GridSearchCV, độ ổn định, đánh giá cuối trên tập kiểm tra
                 06_demo             dùng mô hình cuối dự đoán xe nhập tay, thử "nếu… thì…", thao tác nhanh khi vấn đáp
src/             data.py (đọc, chia tập), preprocess.py (Pipeline), evaluate.py (fold CV, chỉ số),
                 features.py (đặc trưng bổ sung), final_model.py (mô hình cuối), baseline_reference.py
app/             ứng dụng web Streamlit AutoMPG Analytics (Polynomial Regression, Ridge Regression)
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
- Ứng dụng web **AutoMPG Analytics** (Streamlit): `streamlit run app/app.py`, chạy từ thư mục gốc dự án để nhận giao diện trong `.streamlit/config.toml`. Trình duyệt tự mở http://localhost:8501, dừng bằng Ctrl+C. Lần mở đầu tiên mất khoảng 10–20 giây để huấn luyện 180 mô hình, sau đó được lưu cache.
  - Tổng quan: chỉ số dữ liệu, mô hình tốt nhất theo MAE/RMSE/R², phân bố MPG, MPG theo trọng lượng.
  - Khám phá dữ liệu: bảng dữ liệu, tương quan, quan hệ từng đặc trưng với MPG.
  - Dự đoán MPG: nhập thông số xe, chọn Polynomial (bậc 1–10) hoặc Ridge (bậc, α), xem MPG, L/100 km, sai số điển hình, xe tương tự, cảnh báo ngoại suy.
  - Polynomial Regression: bảng bậc 1–10 (Train/Test R², MAE, RMSE, CV RMSE), biểu đồ underfitting/overfitting, minh họa đường cong.
  - Ridge Regression: ảnh hưởng của α, độ lớn hệ số, so sánh với Polynomial ở từng bậc.
  - Mã nguồn: `app/app.py` (điều hướng), `app/modeling.py` (huấn luyện, chạy riêng được: `python app/modeling.py`), `app/core.py` (cache), `app/style.py` (giao diện), `app/views/` (mỗi trang một file).
- Kiểm tra nhanh mô hình cuối: `python src/final_model.py` (in RMSE trên tập kiểm tra, phải là 2,314).
- Mọi thí nghiệm dùng `random_state=42` (hằng `SEED` trong `src/data.py`).
- Dùng đường dẫn tương đối; notebook phải được mở từ thư mục `notebooks/`.
- Thời gian chạy: notebook 03 khoảng 30 giây, 04 khoảng 2 phút, 05 khoảng 5–6 phút (GridSearchCV và CV lặp 5 seed).
- Môi trường đã kiểm tra: Python 3.14, pandas 3.0, scikit-learn 1.9. Bản pandas 2.x vẫn chạy, chỉ khác cách in kiểu dữ liệu chuỗi (`object` thay vì `str`).

## Quy ước chia tập và đánh giá
Chia ~80/20 (318/80), phân tầng theo `origin`, gom theo `name` (cùng tên xe không nằm ở cả hai tập). Mọi so sánh mô hình và chọn siêu tham số dùng cùng 5 fold trên tập huấn luyện (`src/evaluate.py`). Tập kiểm tra chỉ dùng **một lần** ở cuối notebook 05, sau khi đã chốt mô hình.

## Kết quả chính
| Mô hình | RMSE CV 5-fold | RMSE kiểm tra | R² kiểm tra |
|---|---|---|---|
| Tuyến tính 7 đặc trưng (baseline) | 3,33 ± 0,52 | 3,37 | 0,806 |
| Đa thức bậc 2 | 2,87 ± 0,40 | – | – |
| Ridge bậc 6, α = 10 | 2,83 ± 0,39 | – | – |
| Voting (Ridge + Gradient Boosting) – mô hình cuối | 2,60 ± 0,35 (2,69 trung bình 5 seed) | 2,31 | 0,908 |

## Tiến độ
Xem `reports/nhat_ky_dong_gop.md` và file kế hoạch 9 tuần.
