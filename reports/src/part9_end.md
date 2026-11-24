## 12. Hạn chế và hướng phát triển

**Hạn chế của nghiên cứu này**

- Los-loop chỉ 7 ngày: tập test ~1.4 ngày; kết luận về giờ cao điểm dựa trên 3 đợt cao điểm. Cần lặp lại trên METR-LA đầy đủ (4 tháng, cùng 207 cảm biến) hoặc PEMS-BAY (bộ dữ liệu trong danh sách bài báo tham khảo).
- Huấn luyện trên CPU: ≤ 120 epoch với lịch học cosine thay vì 3000 epoch như tác giả; tinh chỉnh và một số biến thể chỉ có 1 seed.
- Không có dữ liệu sự kiện (tai nạn, thời tiết) — nguyên nhân của phần lớn sai số lớn trong giờ cao điểm.

**Hướng phát triển (theo thứ tự lợi ích kỳ vọng / chi phí)**

1. **Đưa vào sản phẩm ngay:** ensemble GRU-RT + bước kiểm tra dữ liệu thiếu (mô hình residual rất nhạy với đầu vào = 0) + cắt dự báo về [0, v_max].
2. **Tích chập đồ thị không làm loãng tín hiệu:** dạng [x, Âx]W đã cho kết quả tốt; bước tiếp theo là diffusion convolution hai chiều của DCRNN (Li et al., 2018 — bài báo thứ 2 trong danh sách tham khảo), phù hợp đồ thị có hướng như SZ-taxi.
3. **Hàm mất mát cho giờ cao điểm:** cân loss theo phương sai cảm biến chưa ổn định (mục 5.4); nên thử trọng số trực tiếp cho mẫu kẹt xe hoặc dự báo phân vị (quantile loss) để có khoảng dự báo đáng tin hơn trong cao điểm.
4. **Conformal theo giờ** để khoảng dự báo đạt độ phủ đúng cả trong giờ cao điểm.
5. **Dữ liệu dài hơn:** với ≥ 1 tháng dữ liệu, đặc trưng thời gian và đồ thị thích nghi có cơ hội phát huy (hiện bị quá khớp).
6. **Điền giá trị thiếu ở đầu vào** (SZ-taxi: số 0 = không có taxi) bằng trung bình láng giềng hoặc hồ sơ theo giờ, thay vì chỉ bỏ qua chúng trong loss (đã thử, không cải thiện).

## 13. Kết luận

Chúng tôi đã cài đặt lại và chạy GRU, GCN, T-GCN cùng 4 baseline cổ điển trên Los-loop và SZ-taxi với cùng dữ liệu, cùng cách chia và 3 seed (≈ 60 run). **GRU tái lập đúng bài báo; T-GCN thì không vượt được GRU** một cách có ý nghĩa trên cả hai bộ dữ liệu — chính Bảng 1 của bài báo cũng cho thấy MAE của GRU tốt hơn. Nguyên nhân cơ chế là phép chuẩn hóa D^(-1/2)(A+I)D^(-1/2) chỉ để lại ~30% trọng số cho tín hiệu của chính nút, làm hỏng dự báo tầm ngắn; bằng chứng đến từ bốn hướng độc lập (thí nghiệm liều–đáp ứng với self-loop, permutation importance, gradient saliency và việc sửa đúng chỗ đó — tách self/láng giềng — làm T-GCN vượt GRU có ý nghĩa thống kê). Con số của bài báo còn được "làm đẹp" bởi việc chọn epoch theo test, cách gộp tầm dự báo và baseline yếu (thiếu persistence; HA trên SZ-taxi yếu gần gấp đôi một trung bình trượt) — trong khi persistence lại là mô hình tốt nhất trong giờ cao điểm.

Cải tiến hiệu quả nhất trên Los-loop không nằm ở đồ thị mà ở **cách đặt bài toán**: học phần thay đổi so với giá trị cuối (residual) và cho mô hình biết giờ trong ngày; ensemble 3 seed của GRU-RT giảm RMSE 7.8% so với GRU và 8.0% so với T-GCN (có ý nghĩa thống kê), vượt mọi con số trong bài báo dù dùng giao thức chặt hơn. Nhưng cải tiến phụ thuộc dữ liệu: trên SZ-taxi (nhiễu, nhiều giá trị thiếu) residual không giúp, và mọi mô hình chỉ hơn trung bình trượt vài phần trăm. Đồ thị có giá trị thật — ở tầm xa, khi tốc độ biến động mạnh, khi dữ liệu nhiễu — nhưng chỉ khi được thiết kế để không xóa mờ thông tin riêng của từng nút.

## Phụ lục A. Cách chạy lại toàn bộ

```bash
pip install -r requirements.txt
python download_data.py --dataset losloop
python download_data.py --dataset sztaxi
python -m unittest discover -s tests -v                   # 8 test
python baselines.py --dataset losloop                     # naive, HA, linear
python baselines.py --dataset sztaxi
python run_suite.py --spec losloop:main losloop:improve losloop:variants losloop:tuning --jobs 4
python run_suite.py --spec losloop:paper sztaxi:paper sztaxi:main sztaxi:improve --jobs 4
python analysis/eda.py
python analysis/compare_models.py --dataset losloop      # bảng chính, ensemble, so sánh bài báo
python analysis/peak_hours.py --dataset losloop
python analysis/tuning_ablation.py --dataset losloop
python analysis/feature_importance.py --dataset losloop
python analysis/explain.py --dataset losloop
python analysis/reliability.py --dataset losloop
python tools/export_models.py                             # models/*.pt (chọn seed theo validation)
python tools/md2docx.py reports/BAO_CAO_TGCN.md reports/BAO_CAO_TGCN.docx
```

Thời gian: ≈ 15–45 phút/run trên 1 nhân CPU; toàn bộ ≈ 60 run ≈ 7–8 giờ với 4 nhân. Mọi run ghi `config.json`, `history.csv` (kèm RMSE test theo epoch chỉ để theo dõi), `metrics.json` (per-step, cumulative, Accuracy/R²/Var), `test_predictions.npz`, `best.pt`.

## Phụ lục B. Cấu trúc mã

| Tệp | Vai trò |
|---|---|
| `tgcn/data.py` | đọc CSV, chuẩn hóa theo train, Â đúng như mã tác giả, cửa sổ trượt, chế độ `--protocol paper` |
| `tgcn/models.py` | GRU, GCN, T-GCN (+ residual, thời gian, đồ thị thích nghi, tách self/láng giềng, Â = I) |
| `tgcn/training.py` | AdamW + cosine, dừng sớm theo validation, loss MSE/Huber/MAE, loss đơn vị gốc, mask số 0 |
| `tgcn/metrics.py` | MAE, RMSE, MAPE, WAPE, Accuracy, R², Var; per-step và cumulative |
| `baselines.py` | persistence, HA cửa sổ, HA theo giờ, hồi quy tuyến tính |
| `train.py`, `evaluate.py`, `predict.py` | huấn luyện một cấu hình; đánh giá checkpoint; nạp mô hình để dự báo |
| `run_suite.py` | toàn bộ thí nghiệm (tiếp tục được khi bị ngắt, chạy song song) |
| `analysis/*.py` | EDA, so sánh, giờ cao điểm, tinh chỉnh/ablation, ảnh hưởng thuộc tính, giải thích, độ tin cậy |
| `models/` | checkpoint cuối cùng dùng cho `predict.py` |

## Tài liệu tham khảo

1. L. Zhao et al., "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction," *IEEE T-ITS*, 2019. arXiv:1811.05320. Mã & dữ liệu: github.com/lehaifeng/T-GCN.
2. Y. Li, R. Yu, C. Shahabi, Y. Liu, "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting," *ICLR*, 2018. arXiv:1707.01926.
3. K. Cho et al., "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation" (GRU), *EMNLP*, 2014.
4. T. Kipf, M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," *ICLR*, 2017.
5. F. X. Diebold, R. S. Mariano, "Comparing Predictive Accuracy," *J. Business & Economic Statistics*, 1995.
6. V. Vovk, A. Gammerman, G. Shafer, *Algorithmic Learning in a Random World* (conformal prediction), 2005.
