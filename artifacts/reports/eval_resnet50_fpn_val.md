# Báo cáo Đánh giá Mô hình Phát hiện Tổn thương Xe

## 1. Các Chỉ số Đánh giá Chính (Key Detection Metrics)

| Metric                   | Giá trị    |
|:-------------------------|:-----------|
| mAP@0.5:0.95 (COCO)      | 0.0001     |
| mAP@0.50 (VOC)           | 0.0008     |
| Precision@0.50           | 0.0013     |
| Recall@0.50              | 0.0426     |
| Mean IoU (Matched)       | 0.5100     |
| Tổng Ground Truths       | 47         |
| Tổng Dự đoán             | 1600       |
| Độ trễ suy luận (ms/ảnh) | 1602.22 ms |
| Tốc độ xử lý (FPS)       | 0.6        |

## 2. Chi tiết Từng Lớp Tổn thương (Per-Class Breakdown)

| Class         |   GT Boxes |   Predictions |   TP (@0.5) |   FP (@0.5) |   Precision |   Recall |   AP@0.50 |   AP@0.5:0.95 |   Mean IoU |
|:--------------|-----------:|--------------:|------------:|------------:|------------:|---------:|----------:|--------------:|-----------:|
| dent          |         13 |            11 |           0 |          11 |      0      |   0      |    0      |        0      |       0    |
| scratch       |         14 |           137 |           0 |         137 |      0      |   0      |    0      |        0      |       0    |
| crack         |          5 |           845 |           0 |         845 |      0      |   0      |    0      |        0      |       0    |
| glass shatter |          1 |             0 |           0 |           0 |      0      |   0      |    0      |        0      |       0    |
| lamp broken   |          7 |           123 |           2 |         121 |      0.0163 |   0.2857 |    0.0047 |        0.0005 |       0.51 |
| tire flat     |          7 |           484 |           0 |         484 |      0      |   0      |    0      |        0      |       0    |

## 3. Kết quả pycocotools COCOeval Chuẩn

| COCO Metric                         |   Giá trị |
|:------------------------------------|----------:|
| AP @[ IoU=0.50:0.95 | area=all ]    |    0      |
| AP @[ IoU=0.50 | area=all ]         |    0      |
| AP @[ IoU=0.75 | area=all ]         |    0      |
| AR @[ IoU=0.50:0.95 | maxDets=100 ] |    0.0001 |
