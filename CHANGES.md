# Thay đổi so với T-GCN gốc và so với phiên bản đầu của dự án

## Trung thành với bài báo / mã tác giả
- Ô T-GCN: một lớp GCN trong mỗi cổng GRU, bias cổng khởi tạo 1.0, đầu ra tuyến tính cho H bước (như `tgcn.py` của tác giả).
- **Ma trận kề chuẩn hóa đúng như `calculate_laplacian` của tác giả**: Â = D^-1/2 (A+I)ᵀ D^-1/2, *không* đối xứng hóa, *cộng thêm* I dù `los_adj.csv` đã có đường chéo 1 (self-loop thực tế = 2). Phiên bản đầu của dự án đặt đường chéo = 1 và đối xứng hóa — giữ lại làm ablation (`variants/tgcn_selfloop1`), cho kết quả kém hơn rõ rệt.
- GRU baseline = cùng ô với Â = I (một bộ trọng số chung cho mọi nút), đúng như baseline của tác giả.
- `--protocol paper` tái lập nguyên vẹn giao thức của tác giả (80/20, chia max toàn chuỗi, cửa sổ test nằm trong 20% cuối, giữ epoch có RMSE test nhỏ nhất) — chỉ dùng để so sánh với Bảng 1.

## Giao thức đánh giá chính (nghiêm ngặt hơn bài báo)
- Chia 70/10/20 theo thời gian; z-score theo từng cảm biến bằng thống kê của train; dừng sớm theo validation; 3 seed.
- Sai số tại đúng bước k (per-step) **và** kiểu gộp 1..k của bài báo (cumulative); thêm Accuracy, R², Var như bài báo; bản bỏ số 0 cho SZ-taxi.
- RMSE test theo epoch được *ghi* (không dùng để chọn) để đo độ lạc quan của việc chọn theo test.

## Mô hình / tùy chọn mới
- `gcn`: baseline GCN 2 lớp (chỉ không gian) như bài báo.
- `--residual`, `--time-features` cho cả GRU và T-GCN; `--self-sep` ([x, Âx]W); `--no-graph` (Â = I); `--raw-loss`; `--loss mae`; `--mask-zero`.
- Baseline cổ điển (`baselines.py`): persistence, HA cửa sổ (đúng định nghĩa của tác giả), HA theo giờ, hồi quy tuyến tính.
- `predict.py`: nạp checkpoint tự chứa (trọng số + cấu hình + thống kê chuẩn hóa), ensemble nhiều checkpoint, cắt dự báo âm.

## Kết luận đã được thực nghiệm kiểm chứng (Los-loop, 3 seed)
- Residual + đặc trưng giờ/thứ: cải thiện có ý nghĩa (GRU-RT 6.98 so với GRU 7.50, p < 0.001).
- Tách self/láng giềng: T-GCN 7.30 so với GRU 7.50 (p = 0.01).
- Không có lợi / không ổn định: đồ thị thích nghi (T-GCN+), loss đơn vị gốc (7.14 ± 0.24), cửa sổ 24 bước, hidden 100. Huber đổi RMSE lấy MAE.
