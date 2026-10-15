# Báo cáo Đánh giá Mô hình yolov8s_seg_cardd_ep40

## 1. Các Chỉ số Tổng thể (Instance Segmentation & Detection)

| Metric | Bounding Box (Box) | Phân đoạn (Mask/Seg) | Ghi chú |
| :--- | :---: | :---: | :--- |
| **mAP@0.50** | **0.7518** | **0.7373** | Chuẩn VOC |
| **mAP@0.5:0.95** | **0.5943** | **0.5619** | Chuẩn COCO đa ngưỡng |

## 2. Thông số Huấn luyện
- **Kiến trúc mô hình**: `yolov8s-seg.pt`
- **Kích thước ảnh**: `640x640`
- **Số epochs**: `40`
- **Batch size**: `8`
- **Checkpoints lưu tại**: `artifacts/checkpoints/yolo/yolov8s_seg_cardd_ep40/weights/best.pt`
