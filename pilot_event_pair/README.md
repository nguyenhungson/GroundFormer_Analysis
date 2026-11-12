# Pilot chấm cặp sự kiện

`event_pair_pilot.py` là lần chạy CPU với GroundFormer Original epoch 12 đóng băng. `event_pair_parser.py` tách vai mốc, vai đích và quan hệ `before/after`. `Ranker.cmp` trong runner triển khai ba cách chấm có cùng số tham số. `Ranker.train_logits` gộp điểm qua các cặp đoạn để học bằng nhãn đáp án.

Runner lịch sử cần đặc trưng CLIP-L TRAIN/VAL, snapshot RoBERTa-base, metadata của checkout upstream, cache baseline và danh sách 2.000 câu đã khóa. Các đường dẫn khai báo ở đầu runner và trong `observable_cue_probe.py`, `observable_cue_audit.py` theo bố cục máy chạy gốc. Ba file `CONFIG.json`, `BASELINE_REFERENCE.npz` và `VAL_REFERENCE_ROWS.json` được `evaluate.reference()` đọc từ cạnh runner. Bộ đề xuất cũ nằm ở `Observable_Cue_Audit_20261109/CPU_2000_V2` trong thư mục dự án. Các mã SHA được kiểm trước khi dùng cache.

Để đọc lại bằng chứng của lần đã chạy, dùng `PILOT_LOCKED.json`, `PILOT_RESULTS.json`, `PILOT_VERIFIED.json` và `PILOT_PER_QUESTION.csv` trong gói Drive. Báo cáo ở [thư mục reports](../reports/BAO_CAO_GROUNDFORMER.md).
