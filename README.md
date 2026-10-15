# Vehicle Damage Detection & Assessment (CarDD)

Hệ thống phát hiện (Object Detection) và phân đoạn mặt nạ (Instance Segmentation) các dạng tổn thương xe cơ giới trên bộ dữ liệu **CarDD**, phục vụ bài toán giám định bồi thường tự động và tích hợp AI Agent.

Hệ thống hỗ trợ 6 loại tổn thương tiêu chuẩn:
- `dent`: Vết móp / biến dạng kim loại
- `scratch`: Vết cọ xát / trầy xước sơn
- `crack`: Vết nứt kính / nứt vỏ cản
- `glass shatter`: Vỡ kính chắn gió / kính cửa
- `lamp broken`: Vỡ cụm đèn chiếu sáng / đèn hậu
- `tire flat`: Nổ lốp / xẹp lốp

---

## 1. Kết quả Đánh giá Thực nghiệm

Được đánh giá trực tiếp trên tập Validation độc lập của CarDD (810 ảnh, 1.744 nhãn tổn thương):

| Mô hình | Input Size | Box mAP@0.5 | Mask mAP@0.5 | Precision | Recall | Latency (RTX 3050) | Kích thước |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Faster R-CNN** (ResNet-50 FPN V1) | 512x512 | 61.7% | ❌ | 38.2% | 68.4% | 56.5 ms (~17 FPS) | 165 MB |
| **Mask R-CNN** (ResNet-50 FPN V2) | 512x512 | 64.9% | 62.4% | 44.8% | 60.1% | 103.0 ms (~9.7 FPS) | 180 MB |
| **YOLOv8s-seg** | 640x640 | 75.2% | 73.7% | **78.4%** | 69.7% | **9.3 ms** (~107 FPS) | **22 MB** |
| **YOLOv8m-seg** | 800x800 | **76.8%** | **76.0%** | 75.9% | **74.4%** | **34.1 ms** (~28 FPS) | 94 MB *(ONNX)* |

> **Khuyến nghị lựa chọn**:
> - **Triển khai Server / Backend AI Service**: Sử dụng **YOLOv8m-seg (800res)** cho độ chính xác cao nhất (mAP 76.8%, Recall 74.4%), đặc biệt nhận diện tốt các vết nứt và xước mảnh.
> - **Triển khai Client / Edge / Real-time**: Sử dụng **YOLOv8s-seg (640res)** (dung lượng chỉ 22MB, đạt 107 FPS trên GPU).

---

## 2. Cài đặt Môi trường

Yêu cầu: Python 3.10+ (đã kiểm thử ổn định trên Python 3.12, PyTorch 2.5.1 + CUDA 12.1).

```bash
# 1. Tạo và kích hoạt môi trường ảo
python -m venv env
.\env\Scripts\activate       # Windows PowerShell
# source env/bin/activate    # Linux / macOS

# 2. Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# 3. Kiểm thử toàn bộ hệ thống
python -m pytest -v
```

---

## 3. Dữ liệu & Chuẩn bị

Bộ dữ liệu gốc đặt trong thư mục `data/CarDD_COCO`:
```text
data/CarDD_COCO/
├── annotations/
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   └── instances_test2017.json
├── train2017/
├── val2017/
└── test2017/
```

Chuyển đổi từ định dạng COCO sang YOLO Segmentation (dùng hardlink, không tốn thêm dung lượng ổ đĩa):
```bash
python scripts/prepare_data.py
```

---

## 4. Huấn luyện Mô hình

### A. Huấn luyện YOLOv8-seg (Khuyến nghị cho Production)

Huấn luyện mới YOLOv8m-seg ở độ phân giải 800x800:
```powershell
python scripts/train_yolo.py `
  --model yolov8m-seg.pt `
  --imgsz 800 `
  --batch 4 `
  --epochs 40 `
  --run-name yolov8m_seg_res800_ep40
```

Tiếp tục huấn luyện (Resume) từ checkpoint bị gián đoạn:
```powershell
python scripts/train_yolo.py --resume
```

Chỉ chạy đánh giá (Validation only) trên checkpoint:
```powershell
python scripts/train_yolo.py `
  --model artifacts/checkpoints/yolo/yolov8m_seg_res800_ep40/weights/best.pt `
  --imgsz 800 `
  --val-only
```

### B. Huấn luyện Mask R-CNN (ResNet-50 FPN V2)
```powershell
python scripts/train.py `
  --config configs/train_maskrcnn.yaml `
  --model-name mask_rcnn `
  --epochs 35
```

---

## 5. Theo dõi Huấn luyện (MLflow)

Toàn bộ siêu tham số, đường cong loss, mAP và artifact được lưu trữ cục bộ tại SQLite database:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Truy cập giao diện tại: `http://localhost:5000`

---

## 6. Xuất Mô hình sang ONNX

Pipeline xuất ONNX đã tích hợp sẵn công cụ `onnxslim` để giản lược đồ thị mạng và kiểm tra tính toàn vẹn qua `onnx.checker`:

```powershell
# Xuất mô hình YOLOv8m 800res sang ONNX
python scripts/export.py `
  --weights artifacts/checkpoints/yolo/yolov8m_seg_res800_ep40/weights/best.pt `
  --imgsz 800 `
  --opset 12
```

File ONNX hoàn chỉnh được tạo tại:
- `artifacts/checkpoints/yolo/yolov8m_seg_res800_ep40/weights/best.onnx` (94.16 MB)
- `artifacts/checkpoints/yolo/yolov8s_seg_cardd_ep40/weights/best.onnx` (40.24 MB)

---

## 7. Chạy Dự đoán (Inference)

### A. Qua dòng lệnh CLI
Chạy dự đoán trên ảnh đơn và tự động lưu kết quả vẽ hộp + mặt nạ:
```powershell
python scripts/predict.py `
  --source data/CarDD_COCO/val2017/000013.jpg `
  --model artifacts/checkpoints/yolo/yolov8m_seg_res800_ep40/weights/best.onnx `
  --conf 0.25 `
  --imgsz 800
```

### B. Tích hợp vào AI Service / Web Agent
Bạn có thể gọi trực tiếp file `.onnx` trong backend Python mà không cần cài đặt PyTorch nặng nề:

```python
from ultralytics import YOLO

# Nạp mô hình ONNX
model = YOLO("artifacts/checkpoints/yolo/yolov8m_seg_res800_ep40/weights/best.onnx", task="segment")

# Dự đoán ảnh xe
results = model.predict(source="car_image.jpg", imgsz=800, conf=0.25)

for r in results:
    for i, box in enumerate(r.boxes):
        label = model.names[int(box.cls.item())]
        confidence = float(box.conf.item())
        coords = box.xyxy[0].tolist()
        
        # Diện tích pixel mặt nạ tổn thương
        mask_pixels = int(r.masks.data[i].sum().item()) if r.masks is not None else 0
        print(f"[{label}] Conf: {confidence:.2f} | Area: {mask_pixels} px")
```

---

## 8. Cấu trúc Thư mục

```text
├── configs/                  # File cấu hình YAML (train, model, YOLO dataset)
├── data/                     # Dữ liệu CarDD (COCO & YOLO format)
├── artifacts/
│   ├── checkpoints/          # Trọng số mô hình (.pt, .pth, .onnx)
│   └── reports/              # Báo cáo đánh giá chi tiết Markdown & JSON
├── scripts/
│   ├── train_yolo.py         # Huấn luyện YOLOv8-seg (có resume, val-only, MLflow)
│   ├── train.py              # Huấn luyện Faster R-CNN & Mask R-CNN
│   ├── export.py             # Xuất ONNX kèm onnxslim & verification
│   ├── predict.py            # Chạy suy luận trên ảnh với ONNX / PyTorch
│   └── prepare_data.py       # Chuyển đổi CarDD COCO sang YOLO
├── src/
│   ├── data/                 # Dataset loader, transforms, mask rasterization
│   ├── models/               # Faster R-CNN & Mask R-CNN module wrappers
│   ├── training/             # Trainer PyTorch (AMP, Gradient Accumulation)
│   ├── evaluation/           # Tính toán mAP@0.5, mAP@0.5:0.95, COCO metrics
│   └── tracking/             # MLflow logging wrapper
└── tests/                    # Bộ kiểm thử tự động pytest
```