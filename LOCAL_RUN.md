# Chạy các thí nghiệm nặng trên máy của bạn

Môi trường của tôi chỉ có CPU 4 nhân, nên một số cấu hình được để lại cho máy có GPU. Mọi lệnh dưới đây **tiếp tục được khi bị ngắt** (bỏ qua run đã có `metrics.json`) và ghi kết quả vào `results/` đúng định dạng mà các script phân tích và `tools/build_report.py` đọc.

## 0. Cài đặt (một lần)

```bash
git clone <repo> && cd Datamining-Btl
git checkout claude/inspiring-hamilton-fy2tai
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# PyTorch có CUDA: chọn đúng lệnh tại https://pytorch.org/get-started/locally/ , ví dụ
# pip install torch --index-url https://download.pytorch.org/whl/cu124
python download_data.py --dataset losloop
python download_data.py --dataset sztaxi
python -c "import torch; print(torch.cuda.is_available())"   # phải in True
python -m unittest discover -s tests -v
```

## 1. Những gì đã chạy xong trong repo (không cần chạy lại)

GRU, GCN, T-GCN, các cải tiến, tinh chỉnh, tái lập giao thức tác giả (3 seed), STID, DCRNN-lite, Graph WaveNet bản gọn (1 seed), LightGBM (3 seed), GRU/T-GCN với hồ sơ theo giờ. Bảng kết quả: `reports/tables/`, báo cáo: `reports/BAO_CAO_TGCN.md`.

## 2. Mô hình cỡ đầy đủ trên GPU (khuyến nghị)

```bash
# Graph WaveNet & STAEformer với kích thước như bài báo gốc, 3 seed, Los-loop
python run_suite.py --spec losloop:modern_gpu --jobs 1 --device cuda --threads 4
# tương tự cho SZ-taxi
python run_suite.py --spec sztaxi:modern_gpu --jobs 1 --device cuda --threads 4
```

| Cấu hình | CPU 4 nhân (đo được) | GPU tầm trung (ước lượng) |
|---|---|---|
| Graph WaveNet `--size full` | ~38 s/epoch | 1–3 s/epoch |
| STAEformer `--size compact` | ~83 s/epoch | 2–4 s/epoch |
| STAEformer `--size full` | ~390 s/epoch | 5–10 s/epoch |

Với GPU có ≥ 8 GB VRAM có thể chạy `--jobs 2`. Nếu thiếu bộ nhớ khi chạy STAEformer full, thêm `--batch-size 16` vào lệnh `train.py` (xem ví dụ dưới).

Một run lẻ:

```bash
python train.py --dataset losloop --model staeformer --size full --residual --horizon 12 --seed 42 --device cuda
python train.py --dataset losloop --model gwnet --size full --horizon 12 --seed 42 --device cuda
```

## 3. Dữ liệu lớn hơn: METR-LA / PEMS-BAY

Los-loop chỉ có 7 ngày (5 ngày train) — quá ít để các mô hình đồ thị/attention phát huy. METR-LA (cùng 207 cảm biến, 4 tháng) và PEMS-BAY (325 cảm biến, 6 tháng) là chuẩn so sánh của DCRNN, Graph WaveNet, STAEformer.

1. Tải theo README của DCRNN (github.com/liyaguang/DCRNN, mục *Data Preparation*): `metr-la.h5`, `pems-bay.h5`, `adj_mx.pkl`, `adj_mx_bay.pkl`. (PEMS-BAY dạng CSV cũng có tại zenodo.org/records/5146275 — nguồn trong Papers.xlsx.)
2. Chuyển sang định dạng của dự án:
   ```bash
   pip install tables
   python tools/prepare_large_dataset.py --name metrla  --speeds metr-la.h5  --adj adj_mx.pkl
   python tools/prepare_large_dataset.py --name pemsbay --speeds pems-bay.h5 --adj adj_mx_bay.pkl
   ```
3. Chạy (METR-LA có số 0 = thiếu dữ liệu ⇒ thêm `--mask-zero`):
   ```bash
   python baselines.py --dataset metrla
   python gbdt.py --dataset metrla --seeds 42 --threads 8 --max-train-rows 2000000
   for m in gru tgcn stid gwnet dcrnn staeformer; do
     python train.py --dataset metrla --model $m --horizon 12 --seed 42 --device cuda --mask-zero --epochs 100 --patience 15
   done
   python train.py --dataset metrla --model gru --residual --time-features --profile --horizon 12 --seed 42 --device cuda --mask-zero
   python analysis/compare_models.py --dataset metrla
   ```
   Tham khảo các con số công bố trên METR-LA (MAE 15/30/60 phút, mục tiêu ≠ 0): DCRNN ≈ 2.77 / 3.15 / 3.60; Graph WaveNet ≈ 2.69 / 3.07 / 3.53; STAEformer ≈ 2.65 / 2.97 / 3.34. Lưu ý các bài đó tính MAE tại đúng bước 3/6/12 và chia 70/10/20 — giống giao thức của dự án này.

## 4. Sau khi chạy xong: cập nhật bảng và báo cáo

```bash
python analysis/compare_models.py --dataset losloop
python analysis/modern_models.py --dataset losloop     # bảng/hình phần mô hình hiện đại
python analysis/reliability.py --dataset losloop --skip-model-runs
python tools/build_report.py && python tools/md2docx.py reports/BAO_CAO_TGCN.md reports/BAO_CAO_TGCN.docx
```

Gửi lại cho tôi (hoặc commit) các thư mục `results/<dataset>/modern_gpu/` nếu muốn tôi viết phần phân tích kết quả GPU vào báo cáo.
