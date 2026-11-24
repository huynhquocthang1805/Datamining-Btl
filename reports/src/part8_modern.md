## 11. Cải tiến nâng cao: kiến thức và công nghệ mới

### 11.1 Ý tưởng — bốn hướng thử

Sau mục 5, mô hình tốt nhất là ensemble GRU-RT (RMSE 6.91 / MAE 4.01) và điểm yếu chung là giờ cao điểm. Chúng tôi thử thêm bốn hướng, **cùng giao thức nghiêm ngặt** (70/10/20, thống kê chỉ từ train, chọn theo validation, 3 seed):

| Hướng | Mô hình | Ý tưởng chính |
|---|---|---|
| A. Kiến trúc hiện đại (2018–2023) | **STID** (Shao et al., 2022) | MLP + embedding "danh tính" của cảm biến, giờ trong ngày, thứ — không đồ thị, không hồi quy |
| | **DCRNN-lite** (Li et al., 2018 — bài thứ 2 trong Papers.xlsx) | GRU với diffusion convolution hai chiều K = 2, *giữ trọng số riêng cho chính nút* (k = 0) |
| | **Graph WaveNet** (Wu et al., 2019) | TCN giãn nở có cổng + GCN khuếch tán + đồ thị thích nghi |
| | **STAEformer** (Liu et al., 2023) | Transformer theo thời gian và theo nút + embedding thích nghi không gian–thời gian |
| B. Khai phá dữ liệu cổ điển | **LightGBM** + đặc trưng không gian–thời gian | ~50 đặc trưng thiết kế tay: độ trễ, thống kê trượt, trung bình láng giềng (Âx), lịch, **hồ sơ tốc độ theo khung giờ của từng cảm biến** (tính từ train); một mô hình cây tăng cường cho mỗi tầm, dự báo phần thay đổi |
| C. Chuyển kiến thức B → học sâu | **GRU-RTP**, **T-GCN-RTSP** | thêm hồ sơ theo giờ của cảm biến làm đầu vào và "thay đổi hồ sơ dự kiến tới t+k" vào đầu ra |
| D. Kết hợp | **Trộn theo từng tầm** | trọng số không âm (tổng 1) cho từng tầm dự báo, **tối ưu trên validation**, rồi mới áp dụng lên test |

STAEformer (83–390 s/epoch trên CPU) và Graph WaveNet cỡ đầy đủ được để chạy trên GPU (`LOCAL_RUN.md`); ở đây Graph WaveNet bản gọn chạy 1 seed.

### 11.2 Kết quả trên Los-loop

MODERN_TABLE

![Hình 20. Xếp hạng mọi mô hình trên cùng tập test Los-loop](figures/modern_ranking_losloop.png)

![Hình 21. Sai số theo tầm dự báo của các mô hình mới](figures/modern_horizon_losloop.png)

Kiểm định cặp so với mô hình tốt nhất của phần trước (ensemble GRU-RT), bootstrap khối 1 giờ + Diebold–Mariano:

MODERN_TESTS

Ba kết luận:

1. **Mô hình tốt nhất mới là trộn theo tầm (RMSE 6.29 / MAE 3.58)** — giảm 9% RMSE so với ensemble GRU-RT (p = 0.001), **16% so với GRU và T-GCN gốc**. LightGBM một mình (6.47 / 3.60) đã hơn mọi mô hình học sâu (p = 0.03 so với ensemble GRU-RT) và cực kỳ ổn định giữa các seed (± 0.001).
2. **Kiến trúc hiện đại "nguyên bản" không thắng trên Los-loop**: STID 7.43 ± 0.21, DCRNN-lite 7.56–7.86 — kém cả ensemble GRU-RT có ý nghĩa thống kê.
3. **Đưa hồ sơ theo giờ vào mạng nơ-ron giúp ở tầm xa** (GRU-RTP 60 phút: 7.95 so với 8.39 của GRU-RT; T-GCN-RTSP là mô hình học sâu tốt nhất, 6.69) nhưng làm tệ đi ở 5 phút và **chưa có ý nghĩa thống kê** so với GRU-RT (p = 0.56 và 0.08).

### 11.3 Phản biện — vì sao các mô hình SOTA không thắng ở đây

Các kiến trúc này được thiết kế và báo cáo trên METR-LA/PEMS-BAY (4–6 tháng). Los-loop chỉ có **5 ngày train** ≈ 1 400 cửa sổ:

- **Số tham số so với dữ liệu:** STID 117 nghìn, DCRNN-lite 67 nghìn, Graph WaveNet 166 nghìn so với 13–14 nghìn của GRU/T-GCN. Embedding thời gian của STID có 288 khung giờ × 32 chiều nhưng mỗi khung chỉ được thấy ~5 lần.
- **Dấu hiệu quá khớp đo được:** epoch tốt nhất của STID là 4–11, DCRNN-lite 25–27, GRU-RTP 6–11, T-GCN-RTSP 6–11 (GRU-RT: 19–37; GRU: 87–101), và độ lệch giữa các seed của STID lớn gấp 30 lần GRU (0.21 so với 0.006).
- **Kết luận đúng là "chưa kiểm chứng được trên dữ liệu đủ lớn", không phải "SOTA kém"**: với METR-LA, Graph WaveNet/STAEformer giảm MAE 60 phút từ ~3.60 (DCRNN) xuống ~3.3–3.5. `tools/prepare_large_dataset.py` + `LOCAL_RUN.md` cho phép chạy lại toàn bộ trên các bộ đó.
- Ngược lại, LightGBM có **dừng sớm riêng cho từng tầm** (số cây tốt nhất giảm từ vài trăm ở 5 phút xuống ~60 ở 60 phút) và cây quyết định khó học thuộc nhiễu hơn — đúng thứ dữ liệu nhỏ cần.

### 11.4 Minh chứng — LightGBM dựa vào thông tin gì? (ảnh hưởng của thuộc tính)

![Hình 22. Đặc trưng quan trọng nhất (trái) và tỉ trọng từng nhóm thông tin theo tầm dự báo (phải)](figures/lgbm_importance_losloop.png)

Tỉ trọng gain theo nhóm thông tin (trung bình 3 seed):

LGBM_GROUPS

- **Hồ sơ theo giờ của chính cảm biến là nguồn thông tin số một** (~50% gain ở 5 phút, ~65% ở 60 phút): độ lệch hiện tại so với hồ sơ (`dev_from_prof`, "đang chậm/nhanh bất thường bao nhiêu") và thay đổi hồ sơ dự kiến tới t+k (`prof_change_k`, "sắp vào/ra giờ cao điểm"). Đây chính là thông tin mô hình học sâu thiếu: sin/cos của giờ cho biết *bây giờ là mấy giờ*, nhưng không cho biết *cảm biến này thường kẹt lúc mấy giờ*.
- **Láng giềng (đồ thị) đóng góp ~11–21%, đạt đỉnh ở tầm 15–25 phút** — khớp với kết quả mục 5 rằng đồ thị có ích nhất ở tầm trung/xa và khi biến động mạnh; ở 5 phút lịch sử của chính nút áp đảo.
- **Trạng thái toàn mạng và bất thường láng giềng (v2) thêm rất ít** (RMSE validation 5.352 → 5.348): thông tin đã nằm trong hồ sơ.
- **Siêu tham số gần như không quan trọng** — mọi cấu hình nằm trong 0.03 RMSE validation:

LGBM_TUNING

### 11.5 Giờ cao điểm: điểm yếu đã được khắc phục

MODERN_SUBSETS

- Lần đầu tiên có mô hình **thắng persistence trong giờ cao điểm ở 15 phút** (LightGBM 4.15, trộn 4.12 so với 4.20) và thắng rất xa ở 60 phút (LightGBM 5.99 so với 7.44 — giảm 19%; GRU-RT 8.02).
- Khi kẹt xe (v < 40 mph), MAE 60 phút giảm từ 13.3 (GRU-RT) xuống 10.4 (LightGBM): mô hình biết rằng cảm biến *đang trong khung giờ thường kẹt* nên không vội dự báo "sắp thông".

### 11.6 Trộn theo tầm — vì sao các thành viên "kém hơn" vẫn có ích

![Hình 23. Trọng số trộn theo tầm dự báo (fit trên validation)](figures/blend_weights_losloop.png)

BLEND_WEIGHTS

Trọng số tối ưu trên validation: LightGBM tăng dần theo tầm (0.55 → 0.75), GRU-RT giảm dần (0.35 → 0.25); **GRU-RTP và T-GCN-RTSP — dù tốt hơn GRU-RT khi đứng một mình — nhận trọng số 0**, vì chúng dùng cùng thông tin hồ sơ như LightGBM nên sai số trùng nhau. Ensemble cần thành viên *đa dạng*, không cần thành viên mạnh nhất. Đây cũng là lý do ensemble LightGBM + GRU-RTP chia đều (6.45) kém hơn trộn có trọng số (6.29).

So với bài báo (cách tính gộp của tác giả): trộn theo tầm đạt PAPER_STYLE_BLEND.

### 11.7 SZ-taxi

SZ_MODERN_SECTION

### 11.8 Mô hình cuối cùng mới và cách đọc lên để dự báo

Theo validation, **LightGBM (v2, 127 lá) là mô hình đơn tốt nhất** và **trộn theo tầm LightGBM + GRU-RT là dự báo cuối cùng** cho Los-loop. Mô hình LightGBM là một thư mục:

| Tệp | Nội dung |
|---|---|
| `model/h01.txt … h12.txt` | 12 mô hình cây tăng cường (một cho mỗi tầm 5…60 phút), định dạng văn bản của LightGBM |
| `model/stats.npz` | trung bình/độ lệch chuẩn train của từng cảm biến, ma trận kề chuẩn hóa, **hồ sơ theo giờ** [288 × 207], danh sách cảm biến, bước thời gian, phiên bản đặc trưng |

Đọc lên và dự báo (cùng lệnh `predict.py`, truyền thư mục thay vì tệp `.pt`):

```bash
python predict.py --checkpoint models/losloop_lgbm --input data/losloop/speed.csv --end-time "2012-03-07 07:30" --compare --sensors 4
```

Quy trình bên trong: đọc 12 mốc gần nhất → chuẩn hóa bằng thống kê train → dựng ~50 đặc trưng (độ trễ, láng giềng qua Â, giờ/thứ, độ lệch so với hồ sơ) → mỗi mô hình h_k dự báo Δ_k → ŷ_{t+k} = x_t + Δ_k → đổi về mph, cắt về ≥ 0. Kết quả dựng lại từ CSV khớp đúng dự báo đã lưu trên test (sai khác ~1e-5 mph).

PREDICT_GBDT_EXAMPLE
