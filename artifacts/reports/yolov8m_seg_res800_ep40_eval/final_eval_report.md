# Báo cáo Đánh giá Mô hình yolov8m_seg_res800_ep40_eval

## 1. Các Chỉ số Tổng thể (Instance Segmentation & Detection)

| Metric | Bounding Box (Box) | Phân đoạn (Mask/Seg) | Ghi chú |
| :--- | :---: | :---: | :--- |
| **mAP@0.50** | **0.7682** | **0.7597** | Chuẩn VOC |
| **mAP@0.5:0.95** | **0.6134** | **0.5820** | Chuẩn COCO đa ngưỡng |

## 2. Thông số
- **Chế độ**: `Đánh giá (Validation only)`
- **Nguồn checkpoint / model**: `artifacts/checkpoints/yolo/yolov8m_seg_res800_ep40/weights/best.pt`
- **Kích thước ảnh**: `800x800`
- **Số epochs**: `35`
- **Batch size**: `4`
- **Checkpoints lưu tại**: `artifacts/checkpoints/yolo/yolov8m_seg_res800_ep40_eval/weights/`
