import os, glob, json, yaml, cv2
from pathlib import Path

# ─── EDIT ME ───────────────────────────────────────────────────────
DATA_YAML = r"C:\Users\LewPyt\Downloads\FYP.v3i.yolov11\data.yaml"    # your existing YAML
OUT_JSON  = Path(DATA_YAML).with_name("test_gt.json")
# ───────────────────────────────────────────────────────────────────

# 1. parse yaml, resolve val directory
with open(DATA_YAML) as f:
    yml = yaml.safe_load(f)
test_dir = os.path.abspath(Path(DATA_YAML) / yml["test"])
names   = yml["names"]                    # list of 50 class names
print("test images dir:", test_dir)

# 2. walk test/images & matching labels to build COCO dict
images, annots = [], []
img_id, ann_id = 1, 1
for img_path in glob.glob(os.path.join(test_dir, "*.*g")):   # .jpg, .png
    h, w = cv2.imread(img_path).shape[:2]
    file = os.path.basename(img_path)
    images.append({"id": img_id, "file_name": file, "height": h, "width": w})

    lbl_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
    if os.path.isfile(lbl_path):
        with open(lbl_path) as f:
            for ln in f:
                cls, x, y, bw, bh = map(float, ln.split())
                x1 = (x - bw/2) * w
                y1 = (y - bh/2) * h
                annots.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls) + 1,  # COCO ids start at 1
                    "bbox": [x1, y1, bw*w, bh*h],
                    "area": bw*bh*w*h,
                    "iscrowd": 0
                })
                ann_id += 1
    img_id += 1

categories = [{"id": i+1, "name": n} for i, n in enumerate(names)]
coco_dict  = {"images": images, "annotations": annots, "categories": categories}

with open(OUT_JSON, "w") as f:
    json.dump(coco_dict, f)
print(f"✓ wrote {OUT_JSON} with {len(images)} images and {len(annots)} boxes")
