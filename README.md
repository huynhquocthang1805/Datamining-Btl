# GRU / T-GCN — Dự báo tốc độ giao thông: tái lập, phản biện và cải tiến

Cài đặt PyTorch độc lập của GRU, GCN, T-GCN (Zhao et al., 2019) và các cải tiến, chạy trên dữ liệu thật **Los-loop** (207 cảm biến, 5 phút) và **SZ-taxi** (156 tuyến, 15 phút) theo ba giai đoạn của đề tài:

1. GRU — kết quả ban đầu, kiểm tra dữ liệu, biểu đồ chuỗi thời gian;
2. T-GCN — cùng dữ liệu, cùng cách chia train/validation/test;
3. So sánh sai số, phân tích giờ cao điểm, thử cải tiến (cửa sổ đầu vào, hidden units, đặc trưng thời gian, residual, cách dùng đồ thị).

📄 **Báo cáo đầy đủ:** [`reports/BAO_CAO_TGCN.md`](reports/BAO_CAO_TGCN.md) (bản Word: `reports/BAO_CAO_TGCN.docx`) — ý tưởng → dẫn dắt → phản biện → minh chứng; ảnh hưởng của thuộc tính; mô hình huấn luyện được và cách nạp để dự báo; độ tin cậy; giải thích cơ chế; so sánh với bài báo.

## Kết quả chính (Los-loop, test 20% cuối, 3 seed, RMSE / MAE trung bình 12 tầm 5…60 phút)

| Mô hình | RMSE | MAE | Ghi chú |
|---|---|---|---|
| Persistence (giữ giá trị cuối) | 8.42 | 4.41 | baseline bài báo không có |
| GRU | 7.50 ± 0.01 | 4.18 ± 0.02 | tái lập đúng bài báo ("15 phút": 5.217 so với 5.218) |
| T-GCN (Â đúng như mã tác giả) | 7.52 ± 0.03 | 4.37 ± 0.04 | **không hơn GRU** (p = 0.76) |
| T-GCN tách self/láng giềng `[x, Âx]W` | 7.30 ± 0.06 | 4.12 ± 0.02 | **hơn GRU có ý nghĩa** (p = 0.01) |
| GRU-RT (residual + giờ/thứ) | 6.98 ± 0.06 | 4.07 ± 0.06 | chọn theo validation |
| **Ensemble 3 seed GRU-RT (mô hình cuối)** | **6.91** | **4.01** | "15 phút" 5.09 / 3.01 < bài báo T-GCN 5.13 / 3.18 |

Phát hiện chính: phép chuẩn hóa D^(-1/2)(A+I)D^(-1/2) của T-GCN chỉ giữ ~30% tín hiệu của chính nút ⇒ dự báo tầm ngắn kém cả persistence; bài báo chọn epoch theo tập test, gộp sai số các tầm, thiếu baseline persistence và có MAE của GRU tốt hơn T-GCN ở cả 8 ô của Bảng 1; trong giờ cao điểm persistence vẫn tốt nhất theo MAE. Chi tiết, kiểm định thống kê và hình trong báo cáo.

## 1. Cài đặt

Python 3.10+.

```bash
pip install -r requirements.txt
```

(Trên Windows: tạo venv bằng `py -m venv .venv` rồi `.venv\Scripts\Activate.ps1`. GPU: cài torch đúng bản CUDA theo pytorch.org. Toàn bộ thí nghiệm trong báo cáo chạy trên CPU 4 nhân.)

## 2. Dữ liệu

```bash
python download_data.py --dataset losloop
python download_data.py --dataset sztaxi
```

Tải từ repo của tác giả (`lehaifeng/T-GCN/data`). Nếu không tải tự động được: tải `los_speed.csv`, `los_adj.csv` (hoặc `sz_*`) và đặt thành `data/<dataset>/speed.csv`, `data/<dataset>/adjacency.csv`. `data/demo/` là dữ liệu **giả lập** chỉ để kiểm thử.

## 3. Kiểm thử nhanh

```bash
python -m unittest discover -s tests -v
python train.py --dataset demo --model tgcn --epochs 2 --max-train-batches 3 --max-val-batches 1 --device cpu
```

## 4. Chạy lại toàn bộ thí nghiệm

```bash
python baselines.py --dataset losloop                 # persistence, HA, HA theo giờ, hồi quy tuyến tính
python baselines.py --dataset sztaxi
python run_suite.py --spec losloop:main losloop:improve losloop:variants losloop:tuning --jobs 4
python run_suite.py --spec losloop:paper sztaxi:paper sztaxi:main sztaxi:improve --jobs 4
```

`run_suite.py` chạy song song và tiếp tục được khi bị ngắt (bỏ qua run đã có `metrics.json`). Nhóm thí nghiệm:

| Nhóm | Nội dung |
|---|---|
| `main` | GRU, GCN, T-GCN × 3 seed (giai đoạn 1–2) |
| `improve` | GRU-RT, T-GCN-RT (+ các biến thể tốt nhất) × 3 seed; SZ-taxi: loss bỏ qua số 0 |
| `variants` | từng thành phần một: +residual, +thời gian, tách self/láng giềng, Â = I, đồ thị thích nghi, Huber, loss đơn vị gốc, self-loop = 1 |
| `tuning` | cửa sổ L ∈ {6, 12, 24}, hidden ∈ {32, 64, 100} |
| `paper` | tái lập đúng giao thức tác giả (`--protocol paper`: 80/20, chuẩn hóa max, chọn epoch theo test) |

Một run lẻ, ví dụ T-GCN-RT tách self/láng giềng:

```bash
python train.py --dataset losloop --model tgcn --self-sep --residual --time-features --horizon 12 --seed 42
```

Tùy chọn của `train.py`: `--model {gru,gcn,tgcn,tgcn_plus}`, `--residual`, `--time-features`, `--self-sep`, `--no-graph`, `--adaptive-graph`, `--loss {mse,huber,mae}`, `--raw-loss`, `--mask-zero`, `--normalization {zscore,max}`, `--protocol {strict,paper}`, `--track-test` (chỉ ghi RMSE test theo epoch để đo độ lạc quan, không dùng để chọn).

## 5. Phân tích và báo cáo

```bash
python analysis/eda.py                                   # giai đoạn 1: kiểm tra dữ liệu, biểu đồ
python analysis/compare_models.py --dataset losloop      # bảng chính, ensemble, so với bài báo, độ lạc quan
python analysis/peak_hours.py --dataset losloop          # giờ cao điểm, kẹt xe, biến động mạnh
python analysis/tuning_ablation.py --dataset losloop     # tinh chỉnh + ablation
python analysis/feature_importance.py --dataset losloop  # ảnh hưởng thuộc tính, saliency
python analysis/explain.py --dataset losloop             # skill, cổng GRU, Δ dự báo, thẻ mô hình
python analysis/reliability.py --dataset losloop         # bootstrap, Diebold–Mariano, nhiễu, conformal
python analysis/paper_protocol.py                        # tái lập giao thức tác giả
python tools/export_models.py                            # models/*.pt (seed chọn theo validation)
python tools/build_report.py                             # reports/BAO_CAO_TGCN.md (bảng điền từ CSV)
python tools/md2docx.py reports/BAO_CAO_TGCN.md reports/BAO_CAO_TGCN.docx
```

Hình ở `reports/figures/`, bảng số liệu ở `reports/tables/`. Mỗi run trong `results/<dataset>/...` có `config.json`, `history.csv`, `metrics.json` (per-step, cumulative kiểu bài báo, Accuracy/R²/Var, bản bỏ số 0 cho SZ-taxi), `test_predictions.npz`, `best.pt`, `learning_curve.png`.

## 6. Dùng mô hình đã huấn luyện để dự báo

```bash
python predict.py --checkpoint models/losloop_gru_res_time_seed*.pt --input data/losloop/speed.csv --end-time "2012-03-07 07:30" --compare
```

Checkpoint tự chứa trọng số, cấu hình và thống kê chuẩn hóa của train; truyền nhiều checkpoint để lấy trung bình (ensemble); dự báo được cắt về ≥ 0. `models/index.json` mô tả từng tệp.

## 7. Cấu trúc

```
tgcn/            data.py (chuẩn hóa, Â như mã tác giả, cửa sổ, --protocol paper) · models.py · training.py · metrics.py · plotstyle.py
baselines.py     train.py · evaluate.py · predict.py · run_suite.py
analysis/        eda · compare_models · peak_hours · tuning_ablation · feature_importance · explain · reliability · paper_protocol
tools/           export_models · build_report · md2docx
models/          checkpoint cuối cùng
reports/         BAO_CAO_TGCN.md/.docx, figures/, tables/
tests/           8 unit test
```

`run_experiments.py`, `compare.py`, `make_report.py`, `run_ablations.py` là công cụ của phiên bản đầu, vẫn chạy được nhưng đã được thay bằng `run_suite.py` và `analysis/`.

## 8. Nguồn và giới hạn

- Zhao et al. (2019), *T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction*, IEEE T-ITS — arXiv:1811.05320; mã và dữ liệu: github.com/lehaifeng/T-GCN.
- Li et al. (2018), *Diffusion Convolutional Recurrent Neural Network* — gợi ý cho cách tích chập đồ thị giữ tín hiệu của chính nút.
- Cài đặt độc lập, không phải mã gốc. Los-loop chỉ 7 ngày (test ≈ 1.4 ngày); huấn luyện ≤ 120 epoch trên CPU (tác giả: 3000). Dữ liệu không có cột thời gian: mốc được dựng từ ngày bắt đầu trong bài báo (đã kiểm chứng bằng hồ sơ giờ cao điểm).
