# Báo cáo Đánh giá Mô hình Phát hiện Tổn thương Xe

## 1. Các Chỉ số Đánh giá Tổng thể (Key Detection Metrics)

| Metric                   | Giá trị   |
|:-------------------------|:----------|
| mAP@0.5:0.95 (COCO)      | 0.4076    |
| mAP@0.50 (VOC)           | 0.6020    |
| Precision@0.50           | 0.1091    |
| Recall@0.50              | 0.8039    |
| Mean IoU (Matched)       | 0.7604    |
| Tổng Ground Truths       | 1744      |
| Tổng Dự đoán             | 12848     |
| Độ trễ suy luận (ms/ảnh) | 110.72 ms |
| Tốc độ xử lý (FPS)       | 9.0       |

## 2. Đánh giá Theo Ngưỡng Độ Tin cậy Vận hành (Operational Confidence Thresholds)

> [!NOTE]
> Ở ngưỡng lọc thấp `score >= 0.05` (chuẩn COCO), mô hình giữ lại nhiều proposal yếu để vẽ trọn vẹn đường cong PR nhằm tính mAP chính xác, dẫn đến Precision thấp (~6.5%) và Recall cao (~89%). Khi ứng dụng thực tế với ngưỡng `score >= 0.50`, Precision tăng mạnh lên ~43.4% và F1-Score đạt ~52.9%.

| Operating Point                 |   Precision |   Recall |   F1-Score | Ghi chú                                              |
|:--------------------------------|------------:|---------:|-----------:|:-----------------------------------------------------|
| Ngưỡng cơ sở (Score >= 0.05)    |      0.1091 |   0.8039 |     0.1922 | Mặc định COCO/VOC để vẽ đường cong PR toàn dải       |
| Ngưỡng cân bằng (Score >= 0.30) |      0.3291 |   0.6101 |     0.4276 | Lọc bớt bóng phản chiếu và gợn sáng nhẹ              |
| Ngưỡng thực tế (Score >= 0.50)  |      0.4679 |   0.5097 |     0.4879 | Khuyến nghị cho pipeline đánh giá bồi thường tự động |

## 3. Chi tiết Từng Lớp Tổn thương (Per-Class Breakdown)

| Class         |   GT Boxes |   Predictions |   TP (s>=0.5) |   FP (s>=0.5) |   Precision (s>=0.5) |   Recall (s>=0.5) |   F1 (s>=0.5) |   AP@0.50 |   AP@0.5:0.95 |   Mean IoU |
|:--------------|-----------:|--------------:|--------------:|--------------:|---------------------:|------------------:|--------------:|----------:|--------------:|-----------:|
| dent          |        501 |          3552 |           199 |           200 |               0.4987 |            0.3972 |        0.4422 |    0.4084 |        0.1894 |     0.7146 |
| scratch       |        728 |          6194 |           341 |           540 |               0.3871 |            0.4684 |        0.4239 |    0.3687 |        0.1691 |     0.7317 |
| crack         |        177 |          2060 |            64 |           146 |               0.3048 |            0.3616 |        0.3307 |    0.2562 |        0.1147 |     0.7133 |
| glass shatter |        135 |           306 |           127 |            30 |               0.8089 |            0.9407 |        0.8699 |    0.959  |        0.7838 |     0.9086 |
| lamp broken   |        141 |           596 |           107 |            79 |               0.5753 |            0.7589 |        0.6544 |    0.7428 |        0.4944 |     0.8487 |
| tire flat     |         62 |           140 |            51 |            16 |               0.7612 |            0.8226 |        0.7907 |    0.8767 |        0.6943 |     0.8965 |
