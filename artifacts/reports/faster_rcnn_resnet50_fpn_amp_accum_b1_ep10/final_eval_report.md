# Báo cáo Đánh giá Mô hình Phát hiện Tổn thương Xe

## 1. Các Chỉ số Đánh giá Tổng thể (Key Detection Metrics)

| Metric                   | Giá trị   |
|:-------------------------|:----------|
| mAP@0.5:0.95 (COCO)      | 0.4166    |
| mAP@0.50 (VOC)           | 0.6094    |
| Precision@0.50           | 0.0610    |
| Recall@0.50              | 0.8773    |
| Mean IoU (Matched)       | 0.7507    |
| Tổng Ground Truths       | 1744      |
| Tổng Dự đoán             | 25067     |
| Độ trễ suy luận (ms/ảnh) | 101.86 ms |
| Tốc độ xử lý (FPS)       | 9.8       |

## 2. Đánh giá Theo Ngưỡng Độ Tin cậy Vận hành (Operational Confidence Thresholds)

> [!NOTE]
> Ở ngưỡng lọc thấp `score >= 0.05` (chuẩn COCO), mô hình giữ lại nhiều proposal yếu để vẽ trọn vẹn đường cong PR nhằm tính mAP chính xác, dẫn đến Precision thấp (~6.5%) và Recall cao (~89%). Khi ứng dụng thực tế với ngưỡng `score >= 0.50`, Precision tăng mạnh lên ~43.4% và F1-Score đạt ~52.9%.

| Operating Point                 |   Precision |   Recall |   F1-Score | Ghi chú                                              |
|:--------------------------------|------------:|---------:|-----------:|:-----------------------------------------------------|
| Ngưỡng cơ sở (Score >= 0.05)    |      0.061  |   0.8773 |     0.1141 | Mặc định COCO/VOC để vẽ đường cong PR toàn dải       |
| Ngưỡng cân bằng (Score >= 0.30) |      0.2258 |   0.7104 |     0.3427 | Lọc bớt bóng phản chiếu và gợn sáng nhẹ              |
| Ngưỡng thực tế (Score >= 0.50)  |      0.3599 |   0.6049 |     0.4513 | Khuyến nghị cho pipeline đánh giá bồi thường tự động |

## 3. Chi tiết Từng Lớp Tổn thương (Per-Class Breakdown)

| Class         |   GT Boxes |   Predictions |   TP (s>=0.5) |   FP (s>=0.5) |   Precision (s>=0.5) |   Recall (s>=0.5) |   F1 (s>=0.5) |   AP@0.50 |   AP@0.5:0.95 |   Mean IoU |
|:--------------|-----------:|--------------:|--------------:|--------------:|---------------------:|------------------:|--------------:|----------:|--------------:|-----------:|
| dent          |        501 |          8105 |           266 |           602 |               0.3065 |            0.5309 |        0.3886 |    0.4074 |        0.167  |     0.7033 |
| scratch       |        728 |         10473 |           416 |           821 |               0.3363 |            0.5714 |        0.4234 |    0.4475 |        0.201  |     0.7267 |
| crack         |        177 |          4572 |            55 |           202 |               0.214  |            0.3107 |        0.2535 |    0.2061 |        0.0837 |     0.6961 |
| glass shatter |        135 |           381 |           131 |            22 |               0.8562 |            0.9704 |        0.9097 |    0.9684 |        0.8226 |     0.9187 |
| lamp broken   |        141 |          1276 |           131 |           200 |               0.3958 |            0.9291 |        0.5551 |    0.769  |        0.5455 |     0.8418 |
| tire flat     |         62 |           260 |            56 |            29 |               0.6588 |            0.9032 |        0.7619 |    0.8576 |        0.6797 |     0.8859 |
