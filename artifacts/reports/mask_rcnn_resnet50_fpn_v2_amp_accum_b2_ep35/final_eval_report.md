# Báo cáo Đánh giá Mô hình Phát hiện Tổn thương Xe

## 1. Các Chỉ số Đánh giá Tổng thể (Key Detection Metrics)

| Metric                   | Giá trị   |
|:-------------------------|:----------|
| mAP@0.5:0.95 (COCO)      | 0.4634    |
| mAP@0.50 (VOC)           | 0.6493    |
| Precision@0.50           | 0.1034    |
| Recall@0.50              | 0.8268    |
| Mean IoU (Matched)       | 0.7742    |
| Tổng Ground Truths       | 1744      |
| Tổng Dự đoán             | 13941     |
| Độ trễ suy luận (ms/ảnh) | 103.05 ms |
| Tốc độ xử lý (FPS)       | 9.7       |

## 2. Đánh giá Theo Ngưỡng Độ Tin cậy Vận hành (Operational Confidence Thresholds)

> [!NOTE]
> Ở ngưỡng lọc thấp `score >= 0.05` (chuẩn COCO), mô hình giữ lại nhiều proposal yếu để vẽ trọn vẹn đường cong PR nhằm tính mAP chính xác, dẫn đến Precision thấp (~6.5%) và Recall cao (~89%). Khi ứng dụng thực tế với ngưỡng `score >= 0.50`, Precision tăng mạnh lên ~43.4% và F1-Score đạt ~52.9%.

| Operating Point                 |   Precision |   Recall |   F1-Score | Ghi chú                                              |
|:--------------------------------|------------:|---------:|-----------:|:-----------------------------------------------------|
| Ngưỡng cơ sở (Score >= 0.05)    |      0.1034 |   0.8268 |     0.1839 | Mặc định COCO/VOC để vẽ đường cong PR toàn dải       |
| Ngưỡng cân bằng (Score >= 0.30) |      0.3103 |   0.6795 |     0.426  | Lọc bớt bóng phản chiếu và gợn sáng nhẹ              |
| Ngưỡng thực tế (Score >= 0.50)  |      0.4475 |   0.6009 |     0.513  | Khuyến nghị cho pipeline đánh giá bồi thường tự động |

## 3. Chi tiết Từng Lớp Tổn thương (Per-Class Breakdown)

| Class         |   GT Boxes |   Predictions |   TP (s>=0.5) |   FP (s>=0.5) |   Precision (s>=0.5) |   Recall (s>=0.5) |   F1 (s>=0.5) |   AP@0.50 |   AP@0.5:0.95 |   Mean IoU |
|:--------------|-----------:|--------------:|--------------:|--------------:|---------------------:|------------------:|--------------:|----------:|--------------:|-----------:|
| dent          |        501 |          4274 |           275 |           378 |               0.4211 |            0.5489 |        0.4766 |    0.4998 |        0.2393 |     0.7332 |
| scratch       |        728 |          6620 |           395 |           617 |               0.3903 |            0.5426 |        0.454  |    0.4482 |        0.2125 |     0.7406 |
| crack         |        177 |          2091 |            82 |           199 |               0.2918 |            0.4633 |        0.3581 |    0.2918 |        0.1376 |     0.7278 |
| glass shatter |        135 |           216 |           129 |             4 |               0.9699 |            0.9556 |        0.9627 |    0.9817 |        0.8556 |     0.9307 |
| lamp broken   |        141 |           598 |           115 |            81 |               0.5867 |            0.8156 |        0.6825 |    0.7857 |        0.5776 |     0.8687 |
| tire flat     |         62 |           142 |            52 |            15 |               0.7761 |            0.8387 |        0.8062 |    0.8885 |        0.758  |     0.9108 |
