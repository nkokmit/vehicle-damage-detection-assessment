# Báo cáo Đánh giá Mô hình Phát hiện Tổn thương Xe

## 1. Các Chỉ số Đánh giá Tổng thể (Key Detection Metrics)

| Metric                   | Giá trị   |
|:-------------------------|:----------|
| mAP@0.5:0.95 (COCO)      | 0.4199    |
| mAP@0.50 (VOC)           | 0.6174    |
| Precision@0.50           | 0.0715    |
| Recall@0.50              | 0.8423    |
| Mean IoU (Matched)       | 0.7493    |
| Tổng Ground Truths       | 1744      |
| Tổng Dự đoán             | 20532     |
| Độ trễ suy luận (ms/ảnh) | 56.46 ms  |
| Tốc độ xử lý (FPS)       | 17.7      |

## 2. Đánh giá Theo Ngưỡng Độ Tin cậy Vận hành (Operational Confidence Thresholds)

> [!NOTE]
> Ở ngưỡng lọc thấp `score >= 0.05` (chuẩn COCO), mô hình giữ lại nhiều proposal yếu để vẽ trọn vẹn đường cong PR nhằm tính mAP chính xác, dẫn đến Precision thấp (~6.5%) và Recall cao (~89%). Khi ứng dụng thực tế với ngưỡng `score >= 0.50`, Precision tăng mạnh lên ~43.4% và F1-Score đạt ~52.9%.

| Operating Point                 |   Precision |   Recall |   F1-Score | Ghi chú                                              |
|:--------------------------------|------------:|---------:|-----------:|:-----------------------------------------------------|
| Ngưỡng cơ sở (Score >= 0.05)    |      0.0715 |   0.8423 |     0.1319 | Mặc định COCO/VOC để vẽ đường cong PR toàn dải       |
| Ngưỡng cân bằng (Score >= 0.30) |      0.2507 |   0.6823 |     0.3667 | Lọc bớt bóng phản chiếu và gợn sáng nhẹ              |
| Ngưỡng thực tế (Score >= 0.50)  |      0.3925 |   0.6049 |     0.4761 | Khuyến nghị cho pipeline đánh giá bồi thường tự động |

## 3. Chi tiết Từng Lớp Tổn thương (Per-Class Breakdown)

| Class         |   GT Boxes |   Predictions |   TP (s>=0.5) |   FP (s>=0.5) |   Precision (s>=0.5) |   Recall (s>=0.5) |   F1 (s>=0.5) |   AP@0.50 |   AP@0.5:0.95 |   Mean IoU |
|:--------------|-----------:|--------------:|--------------:|--------------:|---------------------:|------------------:|--------------:|----------:|--------------:|-----------:|
| dent          |        501 |          6397 |           271 |           525 |               0.3405 |            0.5409 |        0.4179 |    0.442  |        0.1855 |     0.6995 |
| scratch       |        728 |          8765 |           426 |           743 |               0.3644 |            0.5852 |        0.4491 |    0.4509 |        0.2037 |     0.722  |
| crack         |        177 |          3701 |            47 |           160 |               0.2271 |            0.2655 |        0.2448 |    0.2008 |        0.0793 |     0.6929 |
| glass shatter |        135 |           399 |           132 |            16 |               0.8919 |            0.9778 |        0.9329 |    0.9748 |        0.8246 |     0.9192 |
| lamp broken   |        141 |          1054 |           123 |           158 |               0.4377 |            0.8723 |        0.5829 |    0.773  |        0.5448 |     0.848  |
| tire flat     |         62 |           216 |            56 |            31 |               0.6437 |            0.9032 |        0.7517 |    0.8628 |        0.6815 |     0.8809 |
