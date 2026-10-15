# Báo cáo Đánh giá Mô hình Phát hiện Tổn thương Xe

## 1. Các Chỉ số Đánh giá Tổng thể (Key Detection Metrics)

| Metric                   | Giá trị    |
|:-------------------------|:-----------|
| mAP@0.5:0.95 (COCO)      | 0.2576     |
| mAP@0.50 (VOC)           | 0.5194     |
| Precision@0.50           | 0.0938     |
| Recall@0.50              | 0.7500     |
| Mean IoU (Matched)       | 0.6941     |
| Tổng Ground Truths       | 8          |
| Tổng Dự đoán             | 64         |
| Độ trễ suy luận (ms/ảnh) | 6413.45 ms |
| Tốc độ xử lý (FPS)       | 0.2        |

## 2. Đánh giá Theo Ngưỡng Độ Tin cậy Vận hành (Operational Confidence Thresholds)

> [!NOTE]
> Ở ngưỡng lọc thấp `score >= 0.05` (chuẩn COCO), mô hình giữ lại nhiều proposal yếu để vẽ trọn vẹn đường cong PR nhằm tính mAP chính xác, dẫn đến Precision thấp (~6.5%) và Recall cao (~89%). Khi ứng dụng thực tế với ngưỡng `score >= 0.50`, Precision tăng mạnh lên ~43.4% và F1-Score đạt ~52.9%.

| Operating Point                 |   Precision |   Recall |   F1-Score | Ghi chú                                              |
|:--------------------------------|------------:|---------:|-----------:|:-----------------------------------------------------|
| Ngưỡng cơ sở (Score >= 0.05)    |      0.0938 |    0.75  |     0.1667 | Mặc định COCO/VOC để vẽ đường cong PR toàn dải       |
| Ngưỡng cân bằng (Score >= 0.30) |      0.2857 |    0.5   |     0.3636 | Lọc bớt bóng phản chiếu và gợn sáng nhẹ              |
| Ngưỡng thực tế (Score >= 0.50)  |      0.5    |    0.375 |     0.4286 | Khuyến nghị cho pipeline đánh giá bồi thường tự động |

## 3. Chi tiết Từng Lớp Tổn thương (Per-Class Breakdown)

| Class         |   GT Boxes |   Predictions |   TP (s>=0.5) |   FP (s>=0.5) |   Precision (s>=0.5) |   Recall (s>=0.5) |   F1 (s>=0.5) |   AP@0.50 |   AP@0.5:0.95 |   Mean IoU |
|:--------------|-----------:|--------------:|--------------:|--------------:|---------------------:|------------------:|--------------:|----------:|--------------:|-----------:|
| dent          |          1 |            23 |             1 |             2 |               0.3333 |               1   |        0.5    |    0.5    |        0.1    |     0.5826 |
| scratch       |          3 |            34 |             0 |             1 |               0      |               0   |        0      |    0.1114 |        0.0418 |     0.6375 |
| crack         |          1 |             2 |             0 |             0 |               0      |               0   |        0      |    1      |        0.2    |     0.5844 |
| glass shatter |          1 |             1 |             1 |             0 |               1      |               1   |        1      |    1      |        0.8    |     0.8682 |
| lamp broken   |          0 |             3 |             0 |             0 |               0      |               0   |        0      |    0      |        0      |     0      |
| tire flat     |          2 |             1 |             1 |             0 |               1      |               0.5 |        0.6667 |    0.505  |        0.404  |     0.8541 |
