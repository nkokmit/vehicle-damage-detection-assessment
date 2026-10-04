import os
import json
import random
import collections
from pathlib import Path
import cv2
import numpy as np

# 1. Đường dẫn các thư mục
BASE_DIR = Path(__file__).resolve().parent.parent
ANN_FILE = BASE_DIR / "data" / "CarDD_COCO" / "annotations" / "instances_train2017.json"
IMG_DIR = BASE_DIR / "data" / "CarDD_COCO" / "train2017"
OUTPUT_DIR = BASE_DIR / "artifacts" / "eda_gt_100"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Định nghĩa Bảng màu trực quan cho từng Class (BGR cho OpenCV)
CLASS_COLORS = {
    "dent": (255, 140, 0),         # Deep Sky Blue (BGR: 255, 140, 0 -> RGB: 0, 140, 255)
    "scratch": (0, 140, 255),      # Orange (BGR: 0, 140, 255 -> RGB: 255, 140, 0)
    "crack": (0, 215, 255),        # Gold / Yellow (BGR: 0, 215, 255)
    "glass shatter": (211, 0, 148),# Deep Pink / Violet (BGR: 211, 0, 148)
    "lamp broken": (34, 34, 255),  # Crimson Red (BGR: 34, 34, 255)
    "tire flat": (50, 205, 50),    # Lime Green (BGR: 50, 205, 50)
}

# 3. Đọc dữ liệu Annotation
print(f"Loading annotations from: {ANN_FILE}")
with open(ANN_FILE, "r", encoding="utf-8") as f:
    coco_data = json.load(f)

categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
img_dict = {img["id"]: img for img in coco_data["images"]}

# Gom nhóm annotation theo image_id
img_annotations = collections.defaultdict(list)
for ann in coco_data["annotations"]:
    img_annotations[ann["image_id"]].append(ann)

# 4. Lựa chọn 100 ảnh ngẫu nhiên có kiểm soát hạt giống (Seed = 42)
random.seed(42)
all_img_ids = sorted(list(img_dict.keys()))

# Lấy 100 ảnh ngẫu nhiên
sampled_img_ids = sorted(random.sample(all_img_ids, 100))

print(f"Sampled {len(sampled_img_ids)} random images.")

# Thống kê phân bố lớp trong 100 ảnh được chọn
sample_class_counts = collections.Counter()
for iid in sampled_img_ids:
    for ann in img_annotations[iid]:
        cname = categories[ann["category_id"]]
        sample_class_counts[cname] += 1

print("\n--- Distribution of classes in 100 sampled images ---")
for cname, count in sample_class_counts.most_common():
    print(f"  {cname:<15}: {count:>3} bboxes")

# 5. Hàm vẽ Bounding Box và Label lên ảnh
def draw_ground_truth(img, annotations, img_info, sample_idx):
    canvas = img.copy()
    h_img, w_img = canvas.shape[:2]

    # Danh sách các class xuất hiện trong ảnh này
    classes_present = set()

    for ann in annotations:
        cid = ann["category_id"]
        cname = categories[cid]
        classes_present.add(cname)
        color = CLASS_COLORS.get(cname, (0, 255, 0))

        # COCO bbox: [x_min, y_min, width, height]
        x, y, w, h = ann["bbox"]
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = min(w_img - 1, int(round(x + w)))
        y2 = min(h_img - 1, int(round(y + h)))

        # Độ dày nét vẽ tùy biến theo kích thước ảnh
        thickness = max(2, int(min(w_img, h_img) / 350))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        # Chuẩn bị nhãn text
        label_text = f"{cname}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(w_img, h_img) / 1200)
        font_thickness = max(1, int(font_scale * 2))

        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        # Vị trí nền nhãn: đặt phía trên bbox nếu đủ chỗ, nếu sát mép trên thì đặt bên trong bbox
        if y1 - th - baseline - 4 >= 0:
            box_y1 = y1 - th - baseline - 6
            box_y2 = y1
            text_y = y1 - baseline - 3
        else:
            box_y1 = y1
            box_y2 = y1 + th + baseline + 6
            text_y = y1 + th + 2

        box_x1 = x1
        box_x2 = min(w_img - 1, x1 + tw + 8)

        # Vẽ nền nhãn
        cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), color, -1)
        # Vẽ chữ nhãn màu trắng
        cv2.putText(canvas, label_text, (box_x1 + 4, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    # Vẽ Banner thông tin trên đỉnh ảnh
    banner_h = 32
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w_img, banner_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    header_text = f"#{sample_idx:03d} | {img_info['file_name']} ({w_img}x{h_img}) | {len(annotations)} obj: {', '.join(sorted(classes_present))}"
    cv2.putText(canvas, header_text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)

    return canvas, sorted(list(classes_present))

# 6. Xử lý và lưu 100 ảnh
saved_metadata = []
print("\nRendering ground truth bounding boxes...")

for idx, iid in enumerate(sampled_img_ids, 1):
    img_info = img_dict[iid]
    img_file = img_info["file_name"]
    img_path = IMG_DIR / img_file

    if not img_path.exists():
        print(f"Warning: File not found {img_path}")
        continue

    # Đọc ảnh gốc
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Warning: Cannot read {img_path}")
        continue

    anns = img_annotations[iid]
    drawn_img, classes_present = draw_ground_truth(img, anns, img_info, idx)

    # Đặt tên file xuất rõ ràng: 001_000042_scratch_dent.jpg
    class_tag = "_".join(c.replace(" ", "-") for c in classes_present)
    out_filename = f"{idx:03d}_{img_file[:-4]}_{class_tag}.jpg"
    out_path = OUTPUT_DIR / out_filename

    # Lưu ảnh chất lượng cao JPEG (quality 95)
    cv2.imwrite(str(out_path), drawn_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    saved_metadata.append({
        "index": idx,
        "image_id": iid,
        "file_name": img_file,
        "out_file": out_filename,
        "width": img_info["width"],
        "height": img_info["height"],
        "num_objects": len(anns),
        "classes": classes_present,
        "objects": [
            {
                "class": categories[a["category_id"]],
                "bbox": [round(v, 1) for v in a["bbox"]]
            }
            for a in anns
        ]
    })

print(f"Successfully processed and saved {len(saved_metadata)} images to:\n  {OUTPUT_DIR}")

# 7. Xuất file tóm tắt danh sách JSON
meta_file = OUTPUT_DIR / "sample_100_metadata.json"
with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(saved_metadata, f, indent=2, ensure_ascii=False)
print(f"Saved metadata json to: {meta_file}")

# 8. Tạo file HTML Gallery để người dùng mở xem và kiểm tra trực tiếp
html_content = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <title>CarDD - Ground Truth 100 Sample Images</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #121316; color: #e0e0e0; }
    header { background: #1e2025; padding: 16px 24px; border-bottom: 1px solid #2e323b; position: sticky; top: 0; z-index: 100; display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; }
    h1 { font-size: 20px; font-weight: 600; color: #fff; }
    .legend { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
    .badge { padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,0.4); display: inline-block; }
    .bg-dent { background: rgb(0, 140, 255); }
    .bg-scratch { background: rgb(255, 140, 0); }
    .bg-crack { background: rgb(255, 215, 0); color: #111; text-shadow: none; }
    .bg-glass { background: rgb(211, 0, 148); }
    .bg-lamp { background: rgb(255, 34, 34); }
    .bg-tire { background: rgb(50, 205, 50); }
    .filter-btn { padding: 6px 12px; border-radius: 6px; border: 1px solid #444; background: #2a2d35; color: #ccc; cursor: pointer; font-size: 13px; transition: all 0.2s; }
    .filter-btn:hover, .filter-btn.active { background: #3b82f6; color: #fff; border-color: #3b82f6; }
    .container { padding: 20px 24px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 20px; }
    .card { background: #1e2025; border: 1px solid #2e323b; border-radius: 10px; overflow: hidden; transition: transform 0.2s, box-shadow 0.2s; }
    .card:hover { transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.5); border-color: #3b82f6; }
    .card-img-wrap { width: 100%; aspect-ratio: 4/3; background: #000; overflow: hidden; cursor: pointer; }
    .card-img-wrap img { width: 100%; height: 100%; object-fit: contain; transition: transform 0.2s; }
    .card-img-wrap:hover img { transform: scale(1.02); }
    .card-body { padding: 12px 14px; font-size: 13px; }
    .card-title { font-weight: 600; color: #fff; margin-bottom: 6px; display: flex; justify-content: space-between; }
    .card-meta { color: #888; font-size: 12px; margin-bottom: 8px; }
    .card-tags { display: flex; flex-wrap: wrap; gap: 6px; }
    /* Modal view */
    .modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.88); z-index: 1000; justify-content: center; align-items: center; padding: 20px; }
    .modal.open { display: flex; }
    .modal-content { max-width: 95vw; max-height: 92vh; display: flex; flex-direction: column; align-items: center; }
    .modal-img { max-width: 100%; max-height: 84vh; object-fit: contain; border-radius: 6px; }
    .modal-info { color: #fff; margin-top: 10px; font-size: 14px; text-align: center; }
    .modal-nav { position: absolute; top: 50%; transform: translateY(-50%); font-size: 28px; background: rgba(255,255,255,0.15); color: #fff; border: none; padding: 12px 18px; cursor: pointer; border-radius: 8px; user-select: none; }
    .modal-nav:hover { background: rgba(255,255,255,0.3); }
    .modal-nav.prev { left: 24px; }
    .modal-nav.next { right: 24px; }
    .modal-close { position: absolute; top: 20px; right: 24px; font-size: 30px; color: #fff; background: none; border: none; cursor: pointer; }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>CarDD 100 Ground Truth Annotations</h1>
      <p style="font-size: 12px; color: #999; margin-top: 4px;">Tập mẫu 100 ảnh ngẫu nhiên từ CarDD instances_train2017.json</p>
    </div>
    <div class="legend">
      <span class="badge bg-dent">dent</span>
      <span class="badge bg-scratch">scratch</span>
      <span class="badge bg-crack">crack</span>
      <span class="badge bg-glass">glass shatter</span>
      <span class="badge bg-lamp">lamp broken</span>
      <span class="badge bg-tire">tire flat</span>
    </div>
    <div style="display:flex; gap: 8px; flex-wrap: wrap;">
      <button class="filter-btn active" onclick="filterClass('all')">Tất cả (100)</button>
      <button class="filter-btn" onclick="filterClass('dent')">dent</button>
      <button class="filter-btn" onclick="filterClass('scratch')">scratch</button>
      <button class="filter-btn" onclick="filterClass('crack')">crack</button>
      <button class="filter-btn" onclick="filterClass('glass shatter')">glass shatter</button>
      <button class="filter-btn" onclick="filterClass('lamp broken')">lamp broken</button>
      <button class="filter-btn" onclick="filterClass('tire flat')">tire flat</button>
    </div>
  </header>

  <div class="container">
    <div class="grid" id="galleryGrid"></div>
  </div>

  <!-- Modal Lightbox -->
  <div class="modal" id="imageModal" onclick="closeModal(event)">
    <button class="modal-close" onclick="closeModal()">&times;</button>
    <button class="modal-nav prev" onclick="navigateModal(-1); event.stopPropagation();">&#10094;</button>
    <div class="modal-content" onclick="event.stopPropagation()">
      <img id="modalImg" class="modal-img" src="" alt="preview" />
      <div id="modalInfo" class="modal-info"></div>
    </div>
    <button class="modal-nav next" onclick="navigateModal(1); event.stopPropagation();">&#10095;</button>
  </div>

  <script>
    const data = """ + json.dumps(saved_metadata, ensure_ascii=False) + """;
    let currentFilter = 'all';
    let currentModalIdx = 0;
    let visibleList = [...data];

    function getBadgeClass(cname) {
      if (cname === 'dent') return 'bg-dent';
      if (cname === 'scratch') return 'bg-scratch';
      if (cname === 'crack') return 'bg-crack';
      if (cname === 'glass shatter') return 'bg-glass';
      if (cname === 'lamp broken') return 'bg-lamp';
      if (cname === 'tire flat') return 'bg-tire';
      return '';
    }

    function renderGrid() {
      const grid = document.getElementById('galleryGrid');
      grid.innerHTML = '';
      visibleList = currentFilter === 'all' 
        ? data 
        : data.filter(item => item.classes.includes(currentFilter));

      visibleList.forEach((item, idx) => {
        const card = document.createElement('div');
        card.className = 'card';
        card.innerHTML = `
          <div class="card-img-wrap" onclick="openModal(${idx})">
            <img src="${item.out_file}" alt="${item.file_name}" loading="lazy"/>
          </div>
          <div class="card-body">
            <div class="card-title">
              <span>#${item.index} - ${item.file_name}</span>
              <span style="color:#3b82f6;">${item.num_objects} obj</span>
            </div>
            <div class="card-meta">Kích thước: ${item.width} x ${item.height}</div>
            <div class="card-tags">
              ${item.classes.map(c => `<span class="badge ${getBadgeClass(c)}">${c}</span>`).join('')}
            </div>
          </div>
        `;
        grid.appendChild(card);
      });
    }

    function filterClass(cname) {
      currentFilter = cname;
      document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn.innerText.includes(cname) || (cname === 'all' && btn.innerText.includes('Tất cả')));
      });
      renderGrid();
    }

    function openModal(idx) {
      currentModalIdx = idx;
      const item = visibleList[idx];
      document.getElementById('modalImg').src = item.out_file;
      document.getElementById('modalInfo').innerHTML = `<strong>#${item.index} | ${item.file_name}</strong> (${item.width}x${item.height}) - ${item.num_objects} objects: ${item.classes.join(', ')}`;
      document.getElementById('imageModal').classList.add('open');
    }

    function closeModal(e) {
      document.getElementById('imageModal').classList.remove('open');
    }

    function navigateModal(direction) {
      currentModalIdx = (currentModalIdx + direction + visibleList.length) % visibleList.length;
      openModal(currentModalIdx);
    }

    document.addEventListener('keydown', (e) => {
      const modal = document.getElementById('imageModal');
      if (!modal.classList.contains('open')) return;
      if (e.key === 'ArrowLeft') navigateModal(-1);
      if (e.key === 'ArrowRight') navigateModal(1);
      if (e.key === 'Escape') closeModal();
    });

    renderGrid();
  </script>
</body>
</html>
"""

html_file = OUTPUT_DIR / "index.html"
with open(html_file, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Interactive HTML Gallery created at:\n  {html_file}")
print("ALL DONE!")
