# Báo cáo So sánh 2 Mô hình Faster R-CNN trong MLflow

Báo cáo so sánh hai kiến trúc Faster R-CNN ResNet-50 trên tập dữ liệu Vehicle Damage Detection (CarDD COCO format, 7 classes).

## 1. Bảng So sánh Tổng hợp (Summary Table)

| Mô hình                 | Backbone        | Tổng Params   | Trainable Params   | Độ trễ (ms/ảnh)   |   Train Loss |   Best Val Loss |
|:------------------------|:----------------|:--------------|:-------------------|:------------------|-------------:|----------------:|
| faster_rcnn_resnet50_v1 | resnet50_fpn    | 41,324,786    | 41,102,386         | 54.76 ms          |       2.9771 |          2.7004 |
| faster_rcnn_resnet50_v2 | resnet50_fpn_v2 | 43,281,778    | 43,056,434         | 90.41 ms          |       2.1909 |          2.3584 |

## 2. Đặc điểm Kiến trúc 2 Mô hình

* **Model 1: `faster_rcnn_resnet50_v1` (ResNet-50-FPN V1)**:
  * Kiến trúc ResNet-50-FPN cổ điển (Lin et al. 2017).
  * Trọng số nền: `FasterRCNN_ResNet50_FPN_Weights.DEFAULT` (COCO mAP: 37.0).
  * Đầu ROI box predictor: 2 lớp Linear thông thường.
  * Ưu điểm: Ổn định, tiêu tốn ít bộ nhớ GPU, thời gian huấn luyện nhanh.

* **Model 2: `faster_rcnn_resnet50_v2` (ResNet-50-FPN V2)**:
  * Kiến trúc ResNet-50-FPN cải tiến (Torchvision V2 modernization).
  * Trọng số nền: `FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT` (COCO mAP: 46.7, vượt trội +9.7 mAP).
  * Cải tiến: FPN thế hệ mới, tích hợp Synchronized BatchNorm trong RoI heads, anchor generator tối ưu.
  * Ưu điểm: Độ chính xác phát hiện và hội tụ tốt hơn trên các vật thể nhỏ và đa kích thước.

## 3. Xem chi tiết trên Giao diện MLflow UI

Để xem biểu đồ loss curves, siêu tham số và artifacts trực tiếp trên trình duyệt:

```bash
mlflow ui --port 5000
```
Truy cập: [http://localhost:5000](http://localhost:5000) -> Experiment: `vehicle-damage-detection`.
