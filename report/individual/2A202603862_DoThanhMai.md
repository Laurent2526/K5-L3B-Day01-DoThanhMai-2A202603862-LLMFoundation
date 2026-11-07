# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:**2A202603862
- **Họ và tên:**Đỗ Thanh Mai
- **SEED** (in ở ô đầu notebook):423

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

|`Mô hình`	        |F1 test	|F1 SUB 40	|p95 (ms/tin)	|USD/1k tin|
|-------------------|-----------|-----------|---------------|----------|
|BoW + LogReg       |0.867      |0.919	|0.030	|0|
|BoW + MLP	|0.920	|0.923	    |0.474	|0|
|MiniLM + kNN	    |0.888	    |1.000	|141.460	|0|
|MiniLM + LogReg	|0.961	    |0.947	|51.980	|0|
|MiniLM + MLP head	|0.945	    |0.947	|19.739	|0|

## 3. Dự đoán và thực tế

Dự đoán: Tôi cho rằng MiniLM + MLP head sẽ hoạt động tốt hơn MiniLM + kNN vì MLP có khả năng học được ranh giới phân loại từ các embedding.

Thực tế: MiniLM + kNN đạt F1 SUB 1.000, cao hơn MiniLM + MLP head với F1 SUB 0.947.

Nguyên nhân: Tập dữ liệu kiểm tra phụ có thể chứa những tin nhắn gần với các mẫu đã có trong không gian embedding. Vì vậy, phương pháp tìm kiếm láng giềng gần nhất có thể phân loại tốt hơn dự kiến.

## 4. Phân tích 50 case khó

|Model	|F1 trên 50 case khó	|Accuracy|
|-------|-----------------------|--------|
|BoW + LogReg	|0.324	|0.50|
|BoW + MLP	|0.500	|0.60|
|MiniLM + kNN	|0.737	|0.70|
|MiniLM + LogReg	|0.612	|0.62|
|MiniLM + MLP head	|0.627	|0.62|

BoW + LogReg có kết quả thấp nhất với F1 chỉ đạt 0.324. Điều này cho thấy mô hình gặp khó khăn khi xử lý các tin nhắn có cách diễn đạt tự nhiên, ngữ cảnh phức tạp hoặc cố tình che giấu ý định spam. Bag-of-Words chủ yếu dựa trên sự xuất hiện của từ khóa nên không hiểu đầy đủ ngữ nghĩa và mối quan hệ giữa các từ.

Hai ví dụ cụ thể:
- Tin nhắn mời kiếm thêm thu nhập tại nhà, yêu cầu đăng ký bằng đường link.
- Tin nhắn giả danh nhân viên giao hàng, yêu cầu xác nhận thông tin thẻ để nhận bưu kiện.

Cả hai đều có cách diễn đạt giống giao tiếp thông thường nhưng mục đích thực tế là dụ người nhận cung cấp thông tin hoặc truy cập đường link. Mô hình dựa trên từ khóa có thể không nhận diện được ý định này.

## 5. Thử thách cá nhân

Đề bài: Thử nghiệm ảnh hưởng của số lượng láng giềng k đến hiệu quả phân loại của mô hình MiniLM + kNN.

Cách thực hiện: Sử dụng embedding MiniLM đã tạo, lần lượt chạy mô hình với k = 1, k = 5 và k = 25. Sau đó tính F1-score bằng các phép toán NumPy.

Kết quả:
- k = 1: F1 = 0.92
- k = 5: F1 = 0.89
- k = 25: F1 = 0.87
Giải thích: Khi k tăng, mô hình sử dụng nhiều láng giềng hơn để quyết định nhãn. Những láng giềng ít tương đồng hơn có thể làm giảm khả năng phân loại chính xác. Trong thử nghiệm này, k = 1 đạt F1 cao nhất. Tuy nhiên, cần đánh giá trên tập validation để lựa chọn k phù hợp, tránh kết luận chỉ dựa trên tập test.

## 6. Phương án cho sản phẩm thật

Tôi lựa chọn MiniLM + MLP head vì mô hình đạt F1 test 0.945, F1 SUB 0.947 và p95 19.739 ms/tin, đáp ứng yêu cầu p95 dưới 50 ms/tin. So với các mô hình sử dụng Bag-of-Words, MiniLM + MLP head có khả năng hiểu ngữ nghĩa của tin nhắn tốt hơn.

Tuy nhiên, phương án này cần nhiều tài nguyên tính toán và thời gian xử lý cao hơn BoW + Logistic Regression. Vì vậy, có thể sử dụng BoW + Logistic Regression làm bộ lọc ban đầu, sau đó chuyển những tin nhắn có độ tin cậy thấp sang MiniLM + MLP head. Cách tiếp cận này giúp cân bằng giữa tốc độ, chi phí và chất lượng phân loại. Cần đo lại p95 tổng thể sau khi triển khai cơ chế chuyển tiếp.

## 7. AI log và điều đã học

|Lần|	Nội dung hỏi AI	|AI hỗ trợ	     |Tôi đã chỉnh sửa|
|---|----------------------|---------------------|----------------|
|1	|Giải thích cách xây dựng mô hình Logistic Regression|	Hướng dẫn tính sigmoid, loss và gradient descent	|Tự kiểm tra và điều chỉnh code|
|2	|Kiểm tra hàm vectorize	|Phát hiện cách tách từ không đồng nhất	|Thay .split() bằng tokenize()|
|3	|Kiểm tra hàm metrics	|Gợi ý chuyển dữ liệu về NumPy array	|Bổ sung np.asarray()|
|4	|Giải thích cách lựa chọn k trong kNN	|Giải thích ảnh hưởng của số lượng láng giềng	|Tự chạy thử k = 1, 5, 25|
|5	|Phân tích kết quả các mô hình	|Giải thích F1, p95 và đánh đổi hiệu năng	|Đối chiếu với kết quả thực tế|
