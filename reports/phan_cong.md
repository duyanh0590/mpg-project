# Bảng phân công nhiệm vụ

Nhóm 2 sinh viên: **Đào Duy Anh** (DA) và **Lý A Cường** (LC).
Đề tài: Dự đoán mức tiêu hao nhiên liệu của ô tô dựa trên thông số kỹ thuật (Auto MPG).

Nguyên tắc: mỗi người có phần chính của mình ở mỗi tuần; người còn lại đọc và chạy lại phần đó
(đánh dấu "Rà soát chéo") vì vấn đáp sẽ hỏi cả hai người về cả hai phần.
Đóng góp thật được ghi ở `nhat_ky_dong_gop.md` và lịch sử Git; bảng này là phân công dự kiến.

| Tuần | Mốc | Đào Duy Anh | Lý A Cường |
|---|---|---|---|
| 1 | Khởi động | Dựng kho Git, tải dữ liệu, `src/data.py` (chia tập), viết mục 2.1 (phát biểu bài toán, mục tiêu) | Lập từ điển dữ liệu, mục 2.1.3, rà soát Chương 1 và tài liệu tham khảo |
| 2 | EDA | Thống kê mô tả, biểu đồ phân bố và nhận xét (Hình 2.1–2.3) | Tương quan, VIF, biểu đồ phân tán (Hình 2.4–2.6), bảng vấn đề chất lượng dữ liệu |
| 3 | Tiền xử lý | `src/preprocess.py` (Pipeline), điền thiếu, ngoại lai, kiểm tra rò rỉ (notebook 02) | So sánh trước – sau (Hình 2.7–2.8), mục 2.3 và 2.4 (chiến lược, chỉ số đánh giá) |
| 4 | Mô hình | Notebook 03: baseline, hồi quy tuyến tính, đa thức bậc 1–10, đường cong lỗi huấn luyện/kiểm định | Notebook 04: KNN (chọn k), Ridge cơ bản, lý do chọn thuật toán |
| 5 | Đánh giá mô hình | Learning curve, ước lượng bias–variance bằng bootstrap | Bảng so sánh các mô hình với baseline, biểu đồ phần dư, phân tích lỗi |
| 6 | Cải tiến | GridSearchCV (Ridge bậc cao, KNN), không dùng tập kiểm tra để chọn siêu tham số | Cải tiến đặc trưng (chọn đặc trưng, biến đổi log) và thử một phương pháp tổ hợp |
| 7 | Tối ưu và so sánh | Đánh giá cuối trên tập kiểm tra, độ ổn định qua các fold | Bảng hạn chế – giải pháp, ablation trước/sau cải tiến, trực quan hóa kết quả |
| 8 | Hoàn thiện và bảo vệ | README, requirements, chạy lại toàn bộ từ đầu, viết mục 3.1–3.3 | Demo, viết mục 3.4–3.5, danh mục tài liệu tham khảo, slide |

Việc chung mỗi tuần: họp 30 phút đầu tuần, cập nhật `nhat_ky_dong_gop.md` và commit bằng máy của mình.
Việc chung tuần 4, 6, 8, 9: mỗi người đọc lại phần của người kia và tự trả lời được các tiêu chí [CN] của bài.
