# GroundFormer_Analysis

Baseline tái tạo: https://drive.google.com/drive/folders/1KlWgKvLUFN_fhMAxrvXOrVyzu0e7k71g?usp=sharing

Giả thuyết tôi chọn: Với câu hỏi before/after, kiểm đúng vai của sự kiện làm mốc và sự kiện cần tìm có thể giúp GroundFormer chọn đoạn bằng chứng tốt hơn bộ chấm cặp thông thường, khi giữ nguyên đáp án, danh sách đoạn và điều kiện đổi đoạn.

## Báo cáo và mã

- [Hiện tượng quan sát được](reports/HIEN_TUONG_BEFORE_AFTER.md) có bảng VAL và TEST, phân tách lỗi đáp án và lỗi đoạn, cùng bốn ca cụ thể.
- [Báo cáo thảo luận](reports/BAO_CAO_GROUNDFORMER.md) nối hiện tượng với một giả thuyết và kết quả pilot ba seed.
- [Mã GroundFormer gốc](groundformer_original/) được trích từ commit [`96b7e92`](https://github.com/jinhseo/GroundFormer/tree/96b7e92040d8ef65d240ae3c2d135134ae9c7225) của Jinhwan Seo, giữ giấy phép MIT. Dữ liệu NExT-GQA và đặc trưng CLIP-L tải theo hướng dẫn của upstream.
- [Notebook chạy Original](notebooks/GroundFormer_Original_15ep_Train_Val_Test_EveryEpoch_WandB.ipynb) chứa quy trình chạy lại có log theo epoch. TEST từng được theo dõi mỗi epoch trong run này. Báo cáo dùng số TEST mô tả run, còn pilot chỉ dùng TRAIN và VAL.
- [Mã pilot chấm cặp sự kiện](pilot_event_pair/) giữ cấu hình và đường dẫn của lần chạy lịch sử. [Mã phân tích epoch 12](analysis/analyze_epoch12_before_after.py) chạy offline từ kết quả đã lưu.

## Tái tạo bảng before/after

Với Python 3.11 trở lên và `val.csv`, `test.csv`, `gsub_val.json`, `gsub_test.json` từ thư mục `datasets/nextgqa` của commit upstream:

```sh
python analysis/analyze_epoch12_before_after.py --archive EPOCH12_SAVED_OUTPUTS.zip --metadata-dir path/to/upstream/datasets/nextgqa --out output
```

Chương trình đối chiếu câu hỏi theo dòng nguồn và ID công khai, kiểm đáp án và GQA từng dòng, rồi đối chiếu metric tổng hợp với bản lưu. Checkpoint, CSV kết quả và JSON provenance được bàn giao qua Drive.
