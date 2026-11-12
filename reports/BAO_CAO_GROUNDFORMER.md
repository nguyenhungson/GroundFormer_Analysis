# GroundFormer · Từ lỗi quan sát đến giả thuyết và thực nghiệm

**Bản thảo thảo luận nhóm · NExT-GQA · GroundFormer Original epoch 12, seed 666**

## 1. Hiện tượng

Trên VAL, mô hình trả lời đúng **2.030/3.358** câu, nhưng chỉ **621/3.358 (18,49%)** vừa đúng đáp án vừa chọn đoạn đạt IoP. Trong 1.409 câu đúng đáp án mà đoạn trượt, **1.316 câu** có đỉnh đường cong nằm ngoài mọi đoạn bằng chứng được chú giải.

Ở **1.144 câu VAL có `before` hoặc `after`**, QA đạt **683/1.144 (59,70%)**, GQA đạt **155/1.144 (13,55%)**. Trong 528 câu đúng đáp án mà đoạn trượt, **494 câu** có đỉnh ngoài GT. Chẳng hạn, ở câu hỏi người phụ nữ làm gì *trước khi* lau miệng em bé, mô hình trả lời đúng “take napkin” nhưng chọn giây **3,7–11,2**, rời đoạn chú giải **29,2–43,8**. Chú giải này xác định đoạn bằng chứng. Việc mô hình nhầm mốc lau miệng, nhầm sự kiện lấy khăn hay xếp điểm sai vẫn là câu hỏi mở. [File hiện tượng](HIEN_TUONG_BEFORE_AFTER.md)

## 2. Một giả thuyết để thảo luận

**Khi câu hỏi chỉ rõ hai sự kiện bằng `before/after`, điểm số có phân biệt vai “mốc” và “đích” sẽ xếp đoạn bằng chứng tốt hơn điểm số chấm cặp thông thường, trên cùng các đoạn thay thế.**

Ví dụ “lấy khăn trước khi lau miệng” cần tìm mốc *lau miệng*, sau đó xếp các đoạn *lấy khăn* xảy ra trước mốc ấy. Nếu điểm số chỉ cộng bằng chứng “trong hai đoạn có khăn và có lau miệng”, hoán đổi hai vai có thể vẫn cho điểm cao. Bộ chấm đã thử so cùng cặp hai đoạn ở hai cách gán vai. Điểm hơn khi gán đúng vai là tín hiệu dùng để đổi đoạn.

Giả thuyết này dự đoán bộ chấm có đảo vai phải vượt bộ chấm cặp thô và bộ chấm riêng từng sự kiện khi cả ba dùng cùng danh sách đoạn, dữ liệu học và số tham số. Nó áp dụng cho các câu tách được hai sự kiện và có đoạn thay thế phù hợp. Việc có từ `before/after` trong câu chưa xác nhận hai điều kiện này.

## 3. Thực nghiệm đã có

GroundFormer và đáp án của nó được giữ cố định. Bộ chấm bổ sung dùng đặc trưng CLIP-L của các cửa sổ thời gian và RoBERTa-base cho văn bản. Ba nhánh cùng 230.145 tham số, học 2 epoch với 3 seed. TRAIN có 1.193 câu để học và 293 câu để hiệu chỉnh trên video riêng. Nhãn đáp án dùng để học, nhãn thời gian dùng để chấm. Trên 2.000 câu VAL đã khóa, mọi nhánh dùng cùng danh sách đoạn thay thế, giữ hình học đoạn.

| Nhánh | IoP@0,5 / 2.000, seed 666 · 667 · 668 | GQA / 2.000, seed 666 · 667 · 668 |
|---|---:|---:|
| GroundFormer gốc | 589 · 589 · 589 | 388 · 388 · 388 |
| Chấm riêng mốc và đích | 588 · 589 · 589 | 387 · 388 · 388 |
| Chấm cặp thô | 589 · 590 · 589 | 388 · 388 · 388 |
| Chấm cặp có đảo vai | 589 · 591 · 589 | 388 · 390 · 388 |

Nhánh đảo vai tăng trung bình **0,033 điểm IoP@0,5** so với gốc, CI98,33% theo video **[0,000; 0,101]**. So với cặp thô, chênh **+0,017 điểm**, CI **[−0,051; +0,087]**. Hai câu cứu được đều ở seed 667. Một câu cũng được cặp thô cứu, câu còn lại cũng được bộ lọc chỉ dùng thứ tự cứu. Phép thử chưa tách được lợi ích riêng của điểm đảo vai.

Bộ tách áp dụng cho **135/2.000** câu, gồm **132 `after` và 3 `before`**. Chỉ **131** câu có đoạn thay thế, trong đó **14** câu QA đúng, đoạn gốc trượt và có ít nhất một đoạn thay thế đạt IoP. Trần cứu GQA với danh sách và bộ tách đó là **14/2.000 (+0,70 điểm)** nếu giữ được mọi câu đã đúng. Một lần chạy lại ghép cặp 5 epoch chưa có kết quả hoàn chỉnh để so.

**Kết luận để thảo luận.** Việc chọn sai thời điểm là lỗi lớn và đo được. Pilot cho thấy thiết kế đảo vai hiện tại bao phủ rất ít câu, đặc biệt là `before`, và hai ca cứu đều có đối chứng cũng cứu được. Giả thuyết phân biệt vai sự kiện vẫn chưa được xác nhận.
