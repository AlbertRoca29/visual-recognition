"""
Class mapping:
  KITTI 1 (car)        → YOLO 0
  KITTI 2 (pedestrian) → YOLO 1
  KITTI 10 (ignore)    → skipped
"""

import json
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from pycocotools import mask as mask_utils
from tqdm import tqdm


KITTI_TO_YOLO = {1: 0, 2: 1}
KITTI_TO_COCO = {1: 2, 2: 0}
YOLO_CLASS_NAMES = ["car", "pedestrian"]


def rle_to_bbox(rle_str: str, height: int, width: int):
    """Decode KITTI-MOTS RLE → (x1, y1, x2, y2) bounding box."""
    rle = {"size": [height, width], "counts": rle_str.encode()}
    bbox = mask_utils.toBbox(rle)
    x, y, w, h = bbox
    return float(x), float(y), float(x + w), float(y + h)


def bbox_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert absolute xyxy → normalised YOLO xywh."""
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def parse_instances_txt(txt_path: Path):
    # Parse a KITTI-MOTS instances .txt file.

    annotations = defaultdict(list)
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            frame_id = int(parts[0])
            object_id = int(parts[1])
            img_h = int(parts[3])
            img_w = int(parts[4])
            rle_str = parts[5]

            class_id_kitti = object_id // 1000
            if class_id_kitti not in KITTI_TO_YOLO:
                continue  # skip 'ignore' class (10)

            x1, y1, x2, y2 = rle_to_bbox(rle_str, img_h, img_w)
            annotations[frame_id].append((class_id_kitti, x1, y1, x2, y2, img_h, img_w))
    return annotations


def convert_split(
    kitti_root: Path,
    out_root: Path,
    sequences: list[str],
    split_name: str,
):
    img_out = out_root / "images" / split_name
    lbl_out = out_root / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    total_boxes = 0

    for seq in tqdm(sequences, desc=f"[{split_name}] sequences"):
        txt_path = kitti_root / "instances_txt" / f"{seq}.txt"
        if not txt_path.exists():
            print(f"No annotation file for sequence {seq}, skipping.")
            continue

        ann = parse_instances_txt(txt_path)

        img_seq_dir = kitti_root / "training" / "image_02" / seq
        if not img_seq_dir.exists():
            # try testing split
            img_seq_dir = kitti_root / "testing" / "image_02" / seq
        if not img_seq_dir.exists():
            print(f"No image directory for sequence {seq}, skipping.")
            continue

        for img_path in sorted(img_seq_dir.glob("*.png")):
            frame_id = int(img_path.stem)

            # destination filenames  →  <seq>_<frame>.{png,txt}
            stem = f"{seq}_{img_path.stem}"
            dst_img = img_out / f"{stem}.png"
            dst_lbl = lbl_out / f"{stem}.txt"

            # copy image
            shutil.copy2(img_path, dst_img)

            # write label file
            boxes = ann.get(frame_id, [])
            lines = []
            for class_id_kitti, x1, y1, x2, y2, img_h, img_w in boxes:
                yolo_cls = KITTI_TO_YOLO[class_id_kitti]
                cx, cy, bw, bh = bbox_to_yolo(x1, y1, x2, y2, img_w, img_h)
                # clamp to [0,1]
                cx, cy, bw, bh = (
                    max(0.0, min(1.0, cx)),
                    max(0.0, min(1.0, cy)),
                    max(0.0, min(1.0, bw)),
                    max(0.0, min(1.0, bh)),
                )
                if bw > 0 and bh > 0:
                    lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    total_boxes += 1

            with open(dst_lbl, "w") as f:
                f.write("\n".join(lines))

    print(f"[{split_name}] {total_boxes} boxes written.")


def write_dataset_yaml(out_root: Path, coco_eval: bool = False):
    yaml_content = f"""# KITTI-MOTS → YOLO dataset configuration
path: {out_root.resolve()}
train: images/train
val:   images/val

nc: {len(YOLO_CLASS_NAMES)}
names: {YOLO_CLASS_NAMES}

# COCO label mapping (for zero-shot evaluation with COCO-pretrained weights)
# YOLO 0 (car)         ←→ COCO 2
# YOLO 1 (pedestrian)  ←→ COCO 0
"""
    yaml_path = out_root / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"dataset.yaml → {yaml_path}")
    return yaml_path


def write_coco_mapping(out_root: Path):
    mapping = {
        "yolo_to_coco_id": KITTI_TO_COCO,
        "yolo_class_names": YOLO_CLASS_NAMES,
        "coco_class_names": {0: "person", 2: "car"},
        "coco_eval_classes": [0, 2],
    }
    out = out_root / "coco_mapping.json"
    with open(out, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"coco_mapping.json → {out}")


TRAIN_SEQUENCES = [f"{i:04d}" for i in [0, 1, 3, 4, 5, 9, 11, 12, 15, 17, 19, 20]]
VAL_SEQUENCES = [f"{i:04d}" for i in [2, 6, 7, 8, 10, 13, 14, 16, 18]]


def main():
    parser = argparse.ArgumentParser(description="Convert KITTI-MOTS → YOLO format")
    parser.add_argument(
        "--kitti_root",
        type=Path,
        default=Path("/home/mcv/datasets/C5/KITTI-MOTS"),
        help="Path to KITTI-MOTS root directory",
    )
    parser.add_argument(
        "--out_root",
        type=Path,
        default=Path("data/kitti_mots_yolo"),
        help="Output directory for converted dataset",
    )
    parser.add_argument(
        "--train_seqs",
        nargs="+",
        default=TRAIN_SEQUENCES,
        help="Training sequence IDs",
    )
    parser.add_argument(
        "--val_seqs",
        nargs="+",
        default=VAL_SEQUENCES,
        help="Validation sequence IDs",
    )
    args = parser.parse_args()

    print(f"KITTI-MOTS → YOLO converter")
    print(f"Source : {args.kitti_root}")
    print(f"Output : {args.out_root}")

    convert_split(args.kitti_root, args.out_root, args.train_seqs, "train")
    convert_split(args.kitti_root, args.out_root, args.val_seqs, "val")

    yaml_path = write_dataset_yaml(args.out_root)
    write_coco_mapping(args.out_root)

    print(f"\nConversion complete. Dataset yaml: {yaml_path}\n")


if __name__ == "__main__":
    main()
