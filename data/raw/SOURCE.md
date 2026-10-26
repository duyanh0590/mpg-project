# Nguồn dữ liệu

- **Bộ dữ liệu:** Auto MPG, UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/9/auto+mpg). Dữ liệu gốc do Quinlan (1993) dùng trong cuộc thi của Hội Thống kê Mỹ (1983).
- **Bản dùng trong dự án:** `auto-mpg.csv`, tải từ bản sao trong kho seaborn-data: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv (tải ngày 26/10/2026). Trang UCI không truy cập được từ môi trường làm việc nên dùng bản sao này.
- **Khác biệt so với bản UCI:** có tiêu đề cột; `origin` ghi dạng chữ (usa/japan/europe) thay vì 1/2/3; giá trị thiếu của `horsepower` là ô trống thay vì dấu "?".
- **Quy mô:** 398 dòng, 9 cột.
- **Điều kiện sử dụng:** UCI công bố bộ dữ liệu này theo giấy phép CC BY 4.0, cần ghi nguồn khi dùng (nhóm cần xác nhận lại trên trang UCI).
- **Không sửa file này.** Mọi xử lý ghi ra `data/processed/`.
