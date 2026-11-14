# GroundFormer gốc · Các hiện tượng quan sát được

NExT-GQA, checkpoint Original epoch 12, seed 666. Đối chiếu từng dòng của kết quả đã lưu với câu hỏi, 5 đáp án và nhãn thời gian ở snapshot `96b7e92`. VAL có 3.358 câu trên 567 video. TEST có 5.553 câu trên 990 video và đã được theo dõi trong run gốc. `Before` và `after` ở đây là từ xuất hiện trong câu hỏi. Điểm GQA yêu cầu cả đáp án đúng lẫn đoạn dự đoán đạt IoP ≥ 0,5 với ít nhất một đoạn chú giải. [eval_ground.py](https://github.com/jinhseo/GroundFormer/blob/96b7e92040d8ef65d240ae3c2d135134ae9c7225/code/GroundFormer/eval_ground.py)

## 1. Trả lời đúng nhiều hơn định vị đúng

| Tập | QA | GQA | IoP@0,5 | Đỉnh trong GT | mIoU |
|---|---:|---:|---:|---:|---:|
| VAL | 2.030/3.358 (60,45%) | 621/3.358 (18,49%) | 954/3.358 (28,41%) | 1.052/3.358 (31,33%) | 16,91% |
| TEST | 3.381/5.553 (60,89%) | 1.096/5.553 (19,74%) | 1.655/5.553 (29,80%) | 1.834/5.553 (33,03%) | 18,55% |

**Nhận xét.** Trong 1.409 câu VAL đúng đáp án nhưng trượt IoP, có 1.316 câu (93,40%) có đỉnh nằm ngoài mọi đoạn GT. Đây là dấu hiệu trực tiếp để ưu tiên kiểm tra việc chọn thời điểm.

## 2. Nhóm câu có `before/after` mất nhiều điểm ở khâu chọn đoạn

| Tập | Từ trong câu hỏi | Câu / video | QA | GQA | IoP@0,5 | Đỉnh trong GT | mIoU |
|---|---|---:|---:|---:|---:|---:|---:|
| VAL | before | 87 / 75 | 52 (59,77%) | 8 (9,20%) | 13 (14,94%) | 13 (14,94%) | 11,72% |
| VAL | after | 1.057 / 455 | 631 (59,70%) | 147 (13,91%) | 234 (22,14%) | 266 (25,17%) | 16,26% |
| VAL | không có cả hai từ | 2.214 / 552 | 1.347 (60,84%) | 466 (21,05%) | 707 (31,93%) | 773 (34,91%) | 17,42% |
| TEST | before | 155 / 128 | 83 (53,55%) | 25 (16,13%) | 43 (27,74%) | 48 (30,97%) | 18,63% |
| TEST | after | 1.624 / 764 | 959 (59,05%) | 253 (15,58%) | 393 (24,20%) | 451 (27,77%) | 17,54% |
| TEST | cả hai từ | 2 / 2 | 2 (100%) | 0 (0%) | 0 (0%) | 1 (50%) | 16,09% |
| TEST | không có cả hai từ | 3.772 / 966 | 2.337 (61,96%) | 818 (21,69%) | 1.219 (32,32%) | 1.334 (35,37%) | 18,97% |

**Nhận xét.** Trên VAL, gộp hai từ cho QA 59,70% và GQA 13,55%, so với 60,84% và 21,05% ở các câu còn lại. Chênh lệch mô tả ở GQA là 7,50 điểm, lớn hơn chênh 1,14 điểm ở QA. Thứ hạng `before/after` theo GQA đổi giữa VAL và TEST, với nhóm `before` chỉ 87 và 155 câu.

## 3. Hai kiểu lỗi xuất hiện trong cả hai nhóm câu

| Tập | Nhóm | Đáp án đúng, đoạn đạt | Đáp án đúng, đoạn trượt | Đáp án sai, đoạn đạt | Cả hai trượt | Đỉnh ngoài GT trong ô đúng đáp án, trượt đoạn |
|---|---|---:|---:|---:|---:|---:|
| VAL | before | 8 | 44 | 5 | 30 | 44/44 |
| VAL | after | 147 | 484 | 87 | 339 | 450/484 |
| TEST | before | 25 | 58 | 18 | 54 | 54/58 |
| TEST | after | 253 | 706 | 140 | 525 | 663/706 |

**Nhận xét.** Trên VAL có 528 câu chứa hai từ mục tiêu mà đáp án đúng nhưng đoạn trượt. Trong số đó, 494 câu (93,56%) có đỉnh ngoài GT. Ở chiều kia, có 92 câu chọn đoạn đạt IoP nhưng vẫn chọn sai đáp án. Vì vậy cần theo dõi QA và IoP riêng khi thử thay đổi mô hình.

## 4. Nhãn câu hỏi và từ chỉ thời điểm chia dữ liệu khác nhau

Trên VAL, 875/1.057 câu `after` mang nhãn `TN`, còn nhóm `before` có 0/87 câu `TN`. Trên TEST, hai số là 1.351/1.624 và 4/155. Trong riêng `TN`, GQA là 126/879 trên VAL và 214/1.356 trên TEST. Một câu chứa từ chỉ thời điểm có thể là câu hỏi về hành động, lý do hoặc mô tả. Bộ đếm từ cũng bao gồm các câu như “after a while”.

**Nhận xét.** So sánh hai từ đang đồng thời so sánh các nhóm nhãn và dạng câu khác nhau. Nhãn thời gian đánh dấu bằng chứng cho đáp án. Nó chưa cho biết mô hình đã nhận đúng sự kiện làm mốc hay chưa.

## 5. Ví dụ thật trên VAL

| ID | Câu hỏi rút gọn | Đáp án gốc → dự đoán | GT (giây) | Đoạn chọn (giây) | IoP |
|---|---|---|---|---|---:|
| `6329077812_7` | Người phụ nữ làm gì trước khi lau miệng em bé? | take napkin → take napkin | [29,2–43,8] | [3,7–11,2] | 0 |
| `3441428429_10` | Vận động viên nữ làm gì sau khi bạn diễn đặt cô xuống băng? | continue skating → continue skating | [25,1–31,8], [35,0–38,8], [50,2–52,4] | [5,2–10,7] | 0 |
| `8557532213_4` | Cậu bé áo vàng làm gì trước khi chuyền bóng cho cậu áo xanh? | kick it up → look at camera | [0,0–1,7] | [0,7–2,2] | 0,667 |
| `4094488636_4` | Chó nâu làm gì sau khi bước qua dây đỏ? | look around → mount the black dog | [7,1–15,1] | [8,2–13,2] | 1,000 |

**Nhận xét.** Hai ca đầu minh họa đúng đáp án nhưng chọn đoạn hoàn toàn rời GT. Hai ca sau minh họa chọn đoạn đạt nhưng nhận sai hành động. Để quy một ca cho lỗi đảo thứ tự, cần xem video và định danh cả sự kiện làm mốc lẫn sự kiện đích.

## 6. Chín ca xem video ngày 14/11

Các nhận xét dưới đây ghi từ lượt xem video của nhóm và đối chiếu với khung hình đại diện, CSV và đỉnh đường cong đã lưu của Original epoch 12. GT là đoạn bằng chứng được chú giải, còn “đỉnh” là thời điểm có điểm cao nhất trên đường cong đã lưu. Các ca do nhóm chọn để xem lỗi. Những giải thích về nguyên nhân trong bảng là giả thuyết theo từng ca.

| ID | Kết quả so với nhãn VAL | Đoạn GT → đoạn chọn (giây), đỉnh | Nhận xét từ video |
|---|---|---|---|
| `6277800530_3` | Sai QA và đoạn. *Tăng tốc* → *dừng lại*. | 20,9–26,9 → 9,7–15,2. Đỉnh 13,2. | Nước đục và đất ướt có màu gần nhau, cỡ xe trong khung hình cũng thay đổi. Mô hình chọn đoạn sớm hơn cảnh xe ra khỏi nước. Cần tách khả năng nhầm ranh giới đất–nước khỏi ảnh hưởng của góc quay hoặc kích thước vật trong khung. |
| `4075791856_3` | Sai QA và đoạn. *Di chuyển giấy* → *nhìn sang người đàn ông*. | 11,2–14,5 → 19,2–30,2. Đỉnh 24,7. | Cảnh quay xa có nhiều người. Câu hỏi yêu cầu theo dõi tay người phụ nữ trong lúc người đàn ông nói. Khó tách lỗi chọn nhầm người khỏi lỗi nhận nhầm hành động của đúng người. |
| `4075791856_5` | Sai QA và đoạn theo nhãn. *Xoa đầu* → *nghe người đàn ông nói*. IoP 0,173. | 33,8–42,1 → 24,7–35,7. Đỉnh 30,2. | Nhóm nghi đáp án chuẩn cần kiểm lại. Hai mô tả có thể cùng đúng trong một cảnh. Cần xác minh cụ thể người phụ nữ mặc áo đen bắt chéo chân có xoa đầu trong đoạn GT hay không. |
| `2857132732_7` | Đúng QA, sai đoạn. Đáp án *chơi xong với đứa trẻ*. | 23,5–24,0 → 1,2–5,7. Đỉnh 3,7. | Câu hỏi nói rõ `at the end` nhưng đỉnh ở đầu video. Khả năng cụm `at the end` gợi đáp án `finish` là một giả thuyết về đường tắt ngôn ngữ. Một dự đoán đúng chưa đủ để quy cho đường tắt này. |
| `4412508597_6` | Đúng QA, sai đoạn. Đáp án *để quấn dây*. | 0,4–2,9 → 4,2–7,2. Đỉnh 5,7. | Câu hỏi nói `at the start`. Đỉnh đến sau đoạn chú giải, cho thấy một ca lệch thời điểm theo chiều ngược với ca `at the end` ở trên. |
| `4488066715_4` | Đúng QA theo nhãn *gạt tay ra*, sai đoạn. | 0,7–4,7 → 75,2–80,2. Đỉnh 77,7. | Cảnh đầu và cuối đều có em bé, tay người lớn, bát và phới. Ca này gợi khả năng chọn nhầm lần xuất hiện của cảnh. Nhóm có nghi vấn về cách diễn đạt hoặc đối tượng của đáp án, cần duyệt nhãn riêng. |
| `3249402410_3` | Sai QA và đoạn. *Đặt pizza sang bên* → *xoa tay phủi vụn*. | 16,0–20,0 → 9,2–14,2. Đỉnh 12,2. | Cần phân biệt động tác tay của người mặc áo đen bên phải và bám mốc *rải phô mai xong*. Dự đoán sai chưa cho biết lỗi nằm ở nhận động tác, xác định đúng người hay tìm mốc. |
| `5116088152_1` | Sai QA và đoạn. *Để giữ chặt trái cây* → *để kéo trái cây ra*. | 5,1–8,7 → 19,7–25,7. Đỉnh 22,2. | Có nhiều lượt tương tác tay–miệng–trái cây. Nhóm nghi mô hình gán nhầm vật hoặc nhầm lần tương tác. Trong nguyên văn, `other` bổ nghĩa cho *tay còn lại* (`other hand`), nên giữ riêng giả thuyết nhầm vật với việc đọc câu hỏi. |
| `5116088152_7` | Sai QA và đoạn. *Nắm lấy quả chuối* → *cười*. | 18,6–27,0 → 29,2–35,7. Đỉnh 33,2. | Câu yêu cầu tìm động tác chỉ tay của người lớn, rồi đọc phản ứng tay của em bé. Đây là ca cần xem chuyển động ngắn liên tiếp để phân biệt bỏ lỡ mốc chỉ tay với bỏ lỡ động tác nắm chuối. |

**Nhận xét.** Trong chín ca đã chọn, sáu ca sai cả đáp án và đoạn theo nhãn, ba ca đúng đáp án nhưng sai đoạn. Cả chín có đỉnh ngoài GT. Chỉ một ca có đoạn dự đoán giao GT (`4075791856_5`). Các ca `2857132732_7` và `4412508597_6` cho thấy mô hình có thể chọn lệch dù câu hỏi cho sẵn mốc đầu/cuối. Các ca còn lại đặt ra các vấn đề khác nhau về đối tượng, động tác, mốc sự kiện hoặc độ rõ của nhãn. Mẫu này chưa đo được tần suất của từng nguyên nhân trên VAL.

**Âm thanh.** Baseline Original epoch 12 nhận đặc trưng ảnh từng khung hình từ CLIP-L và văn bản câu hỏi/đáp án. Đường chạy này không nhận audio. Với hai câu `4075791856`, “người đàn ông đang nói” vì vậy là mốc mô hình phải suy ra từ hình ảnh và văn bản. Điều này giải thích nguồn tín hiệu mà mô hình có thể dùng cho mốc, còn nguyên nhân sai của từng câu vẫn cần kiểm. [Mã trích đặc trưng ảnh](../groundformer_original/code/GroundFormer/tools/extract_feat_vid.py#L91-L116) · [Mã suy luận](../groundformer_original/code/GroundFormer/train/trainval_gdqa.py#L60-L93)
