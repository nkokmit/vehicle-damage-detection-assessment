# Báo cáo Đánh giá Mô hình yolov8m_seg_res800_ep40

## 1. Các Chỉ số Tổng thể (Instance Segmentation & Detection)

| Metric | Bounding Box (Box) | Phân đoạn (Mask/Seg) | Ghi chú |
| :--- | :---: | :---: | :--- |
| **mAP@0.50** | **0.7681** | **0.7580** | Chuẩn VOC |
| **mAP@0.5:0.95** | **0.6128** | **0.5765** | Chuẩn COCO đa ngưỡng |

## 2. Thông số
- **Chế độ**: `Tiếp tục huấn luyện (Resume)`
- **Nguồn checkpoint / model**: `artifacts\checkpoints\yolo\yolov8m_seg_res800_ep40\weights\last.pt`
- **Kích thước ảnh**: `640x640`
- **Số epochs**: `35`
- **Batch size**: `8`
- **Checkpoints lưu tại**: `artifacts/checkpoints/yolo/yolov8m_seg_res800_ep40/weights/`
