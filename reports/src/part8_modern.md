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

STAEformer (83–390 s/epoch trên CPU) và Graph WaveNet cỡ đầy đủ được để chạy trên GPU (`LOCAL_RUN.md`); ở đây Graph WaveNet bản gọn chạy 1 seed không residual và 3 seed có residual (≈ 30–50 phút/run trên CPU).

### 11.2 Kết quả trên Los-loop

MODERN_TABLE

![Hình 20. Xếp hạng mọi mô hình trên cùng tập test Los-loop](figures/modern_ranking_losloop.png)

![Hình 21. Sai số theo tầm dự báo của các mô hình mới](figures/modern_horizon_losloop.png)

Kiểm định cặp so với mô hình tốt nhất của phần trước (ensemble GRU-RT), bootstrap khối 1 giờ + Diebold–Mariano:

MODERN_TESTS

Bốn kết luận:

1. **Mô hình tốt nhất mới là trộn theo tầm LightGBM + Graph WaveNet-residual (+ GRU-RT): RMSE 6.19 / MAE 3.49** — giảm 10% RMSE so với ensemble GRU-RT (p = 0.0002), **18% RMSE và 20% MAE so với T-GCN gốc**, 17% / 16% so với GRU.
2. **LightGBM với đặc trưng thiết kế tay là mô hình đơn tốt nhất (6.47 / 3.60)**, cực kỳ ổn định giữa các seed (± 0.001), hơn ensemble GRU-RT có ý nghĩa (p = 0.03).
3. **Kiến trúc hiện đại: kết quả phân hóa.** **Graph WaveNet + residual (6.59 ± 0.12 / 3.67)** là mô hình đơn tốt thứ hai và là mạng nơ-ron tốt nhất (p = 0.003 so với ensemble GRU-RT) — nhưng *không có residual* thì chỉ 6.86 (không khác biệt có ý nghĩa). STID (7.43 ± 0.21) và DCRNN-lite (7.56–7.86) kém hơn ensemble GRU-RT có ý nghĩa.
4. **Đưa hồ sơ theo giờ vào mạng nơ-ron giúp ở tầm xa** (GRU-RTP 60 phút: 7.95 so với 8.39 của GRU-RT; T-GCN-RTSP 6.69) nhưng làm tệ đi ở 5 phút và **chưa có ý nghĩa thống kê** so với GRU-RT (p = 0.56 và 0.08).

### 11.3 Phản biện — vì sao các mô hình SOTA không thắng ở đây

Các kiến trúc này được thiết kế và báo cáo trên METR-LA/PEMS-BAY (4–6 tháng). Los-loop chỉ có **5 ngày train** ≈ 1 400 cửa sổ:

- **Số tham số so với dữ liệu:** STID 117 nghìn, DCRNN-lite 67 nghìn, Graph WaveNet 166 nghìn so với 13–14 nghìn của GRU/T-GCN. Embedding thời gian của STID có 288 khung giờ × 32 chiều nhưng mỗi khung chỉ được thấy ~5 lần.
- **Dấu hiệu quá khớp đo được:** epoch tốt nhất của STID là 4–11, DCRNN-lite 25–27, GRU-RTP 6–11, T-GCN-RTSP 6–11 (GRU-RT: 19–37; GRU: 87–101), và độ lệch giữa các seed của STID lớn gấp 30 lần GRU (0.21 so với 0.006).
- **Ngoại lệ có tính hướng dẫn — Graph WaveNet + residual:** cũng quá khớp sớm (epoch tốt nhất 10–13) nhưng vẫn tốt: tích chập thời gian giãn nở + khuếch tán đồ thị hai chiều + đồ thị thích nghi cho biểu diễn đủ mạnh, còn residual (học Δ so với giá trị cuối — phát hiện ở mục 5.4) giữ cho mô hình không bao giờ tệ hơn persistence ở tầm ngắn. Bỏ residual: 6.86. Mẹo "đặt bài toán đúng" quan trọng ngang kiến trúc.
- **Kết luận đúng là "chưa kiểm chứng được trên dữ liệu đủ lớn", không phải "SOTA kém"** (mục 11.7 cho thấy STID dẫn đầu trên SZ-taxi nhiều dữ liệu hơn): với METR-LA, Graph WaveNet/STAEformer giảm MAE 60 phút từ ~3.60 (DCRNN) xuống ~3.3–3.5. `tools/prepare_large_dataset.py` + `LOCAL_RUN.md` cho phép chạy lại toàn bộ trên các bộ đó.
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

- Lần đầu tiên có mô hình **thắng persistence trong giờ cao điểm ở 15 phút** (LightGBM 4.15, trộn 4.04 so với 4.20) và thắng rất xa ở 60 phút (trộn 5.88, LightGBM 5.99 so với 7.44 của persistence — giảm 19–21%; GRU-RT 8.02).
- Khi kẹt xe (v < 40 mph), MAE 60 phút giảm từ 13.3 (GRU-RT) xuống 10.4 (LightGBM): mô hình biết rằng cảm biến *đang trong khung giờ thường kẹt* nên không vội dự báo "sắp thông".

### 11.6 Trộn theo tầm — vì sao thành viên "kém hơn" vẫn có ích

![Hình 23. Trọng số trộn theo tầm dự báo (fit trên validation)](figures/blend_weights_losloop.png)

BLEND_WEIGHTS

Trọng số tối ưu trên validation chia gần đều cho **LightGBM (~0.40) và Graph WaveNet-residual (~0.50)**, GRU-RT nhận ~0.05, T-GCN-RTSP gần 0. Hai mô hình đơn tốt nhất ở đây thuộc **hai họ rất khác nhau** — cây quyết định trên đặc trưng thiết kế tay và mạng nơ-ron đồ thị học biểu diễn trực tiếp — nên sai số của chúng ít tương quan và trộn lại giảm thêm 4–6% RMSE so với từng mô hình.

Bản trộn trước khi có Graph WaveNet (LightGBM + GRU-RT + GRU-RTP + T-GCN-RTSP) cho 6.29 / 3.58, với LightGBM 0.55 → 0.75 (tăng theo tầm) và GRU-RT 0.35 → 0.25; khi đó **GRU-RTP và T-GCN-RTSP — dù tốt hơn GRU-RT khi đứng một mình — nhận trọng số 0**, vì chúng dùng cùng thông tin hồ sơ như LightGBM nên sai số trùng nhau. Ensemble cần thành viên *đa dạng*, không cần thành viên mạnh nhất; vì vậy ensemble chia đều LightGBM + GRU-RTP (6.45) kém hơn trộn có trọng số.

So với bài báo (cách tính gộp của tác giả): mô hình trộn đạt PAPER_STYLE_BLEND.

### 11.7 SZ-taxi

Cùng các hướng trên SZ-taxi (22 ngày train, mục tiêu ≠ 0; Graph WaveNet/DCRNN không chạy trên SZ-taxi vì giới hạn CPU):

SZ_MODERN_TABLE

![Hình 24. SZ-taxi: xếp hạng mọi mô hình (mục tiêu ≠ 0)](figures/modern_ranking_sztaxi.png)

Trên SZ-taxi, bức tranh **đảo ngược** so với Los-loop — và chính sự đảo ngược đó là minh chứng:

- **STID là mô hình đơn tốt nhất (4.445 ± 0.009 / 3.140)**, hơn LightGBM v2 (4.494) và GRU (4.619); mọi khác biệt so với ensemble GRU-RT đều có ý nghĩa (p < 10⁻⁴). Trộn theo tầm (STID 0.75–0.80, LightGBM 0.10–0.20) đạt **4.402 / 3.106**, giảm 4.7% RMSE và 5.0% MAE so với GRU.
- **Vì sao STID thắng ở đây mà thua ở Los-loop?** Embedding "danh tính cảm biến" và "khung giờ" của STID thực chất **tự học hồ sơ theo giờ của từng cảm biến** — đúng tri thức mà LightGBM nhận qua đặc trưng thiết kế tay. Với SZ-taxi mỗi khung giờ được thấy ~22 lần trong train, đủ để học; với Los-loop chỉ ~5 lần, nên STID quá khớp. Cùng một ý tưởng, hai kết luận ngược nhau — phụ thuộc vào lượng dữ liệu.
- LightGBM v2 (thêm bất thường láng giềng/toàn mạng) cải thiện rõ trên SZ-taxi (4.494 so với 4.523 của v1), khác với Los-loop: đặc trưng trạng thái toàn mạng hữu ích hơn trên chuỗi nhiễu, nơi trung bình hóa nhiều cảm biến lọc được nhiễu.
- Residual tiếp tục không giúp trên dữ liệu nhiễu (STID + residual 4.449 ≈ STID 4.445), nhất quán với mục 10.

### 11.8 Mô hình cuối cùng mới và cách đọc lên để dự báo

Theo validation, **LightGBM (v2, 127 lá) là mô hình đơn tốt nhất**, **Graph WaveNet + residual là mạng nơ-ron tốt nhất**, và **trộn theo tầm hai mô hình này (+ GRU-RT với trọng số nhỏ) là dự báo cuối cùng** cho Los-loop. Các tệp trong `models/`: `losloop_lgbm/` (LightGBM), `losloop_gwnet_res.pt` (Graph WaveNet, seed tốt nhất theo validation), `losloop_gru_res_time_seed*.pt` (GRU-RT); SZ-taxi: `sztaxi_lgbm/`, `sztaxi_stid.pt`. Mô hình LightGBM là một thư mục:

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

So với ví dụ cùng thời điểm ở mục 6.2: với cảm biến đang kẹt 767542 (thực tế 18–23 mph), ensemble GRU-RT dự báo "hồi phục" dần lên 38.5 mph sau 1 giờ, còn **LightGBM giữ ở 22–27 mph** — mô hình biết cảm biến này đang ở trong khung giờ thường kẹt (đặc trưng `dev_from_prof`, `prof_change_k`), nên không kéo dự báo về mức thông thoáng. Đây là minh chứng cụ thể cho việc giảm MAE khi kẹt xe ở mục 11.5.

Graph WaveNet được nạp đúng như GRU/T-GCN (`predict.py --checkpoint models/losloop_gwnet_res.pt ...`, checkpoint chứa kiến trúc, trọng số, ma trận chuyển P_f, P_b và thống kê chuẩn hóa). Trộn theo tầm không cần mã riêng: lấy dự báo của LightGBM, Graph WaveNet và GRU-RT rồi cộng có trọng số theo bảng ở mục 11.6 (trọng số lưu trong `results/losloop/ensemble/blend/metrics.json`).
