# detr_inference.py
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DetrImageProcessor, DetrForObjectDetection
from tqdm import tqdm
from pycocotools import mask as maskUtils

from config import TRAIN_IMG_DIR, INSTANCES_DIR
from utils import map_label_to_class, load_annotations

# GT_FILE = "gt.txt"
PRED_FILE = "detr_predictions.txt"

# ---------------- Model ----------------
MODEL_NAME = "facebook/detr-resnet-50"
CONF_THRESH = 0.7
PCT_VIS = 0.02
BATCH_SIZE = 16  # adjust based on GPU memory

# ---------------- Dataset ----------------
class ImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(str(path))[:, :, ::-1].copy()  # BGR -> RGB
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # C,H,W
        return img, str(path)

# ---------------- Group images by sequence ----------------
images_by_seq = defaultdict(list)
for child in sorted(TRAIN_IMG_DIR.iterdir()):
    if child.is_dir():
        for f in sorted(child.iterdir()):
            if f.suffix.lower() == ".png":
                images_by_seq[child.name].append(f)

# ---------------- Model Setup ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

processor = DetrImageProcessor.from_pretrained(MODEL_NAME)
model = DetrForObjectDetection.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ---------------- Output Files ----------------
summary = {"processed": 0, "predicted_total": 0, "gt_total": 0}
sequences_cache = {}

with open(PRED_FILE, "w") as pf:#, open(GT_FILE, "w") as gf:
    pf.write("# seq_name frame track_id class_id x y w h score\n")
    # gf.write("# seq_name frame track_id class_id x y w h\n")

    for seq_name, image_paths in images_by_seq.items():
        dataset = ImageDataset(image_paths)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8, pin_memory=True, shuffle=False)

        # Load GT annotations once per sequence
        inst_file = INSTANCES_DIR / f"{seq_name}.txt"
        if inst_file.exists() and seq_name not in sequences_cache:
            sequences_cache[seq_name] = load_annotations(inst_file)

        for batch_imgs, batch_paths in tqdm(loader, desc=f"Processing {seq_name}"):
            batch_imgs = batch_imgs.to(device)
            sizes = [(img.shape[1], img.shape[2]) for img in batch_imgs]

            # Preprocess images
            inputs = processor(
                images=[(img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8) for img in batch_imgs],
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                from torch.cuda.amp import autocast
                with autocast():
                    outputs = model(**inputs)

            target_sizes = torch.tensor(sizes, device=device)
            detections = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=CONF_THRESH
            )

            for img_tensor, img_path, det in zip(batch_imgs, batch_paths, detections):
                pred_boxes, pred_labels, pred_scores = [], [], []

                # Predictions
                for score, label, box in zip(
                    det["scores"].cpu().numpy(),
                    det["labels"].cpu().numpy(),
                    det["boxes"].cpu().numpy()
                ):
                    x0, y0, x1, y1 = box
                    x, y, w, h = float(x0), float(y0), float(x1 - x0), float(y1 - y0)
                    label_name = model.config.id2label[int(label)]
                    class_id = map_label_to_class(label_name)
                    if class_id is None:
                        continue
                    pred_boxes.append([x, y, w, h])
                    pred_labels.append(f"{label_name}:{score:.2f}")
                    pred_scores.append(float(score))

                # Ground truth
                gt_boxes = []
                frame_id = int(Path(img_path).stem)
                objs = sequences_cache.get(seq_name, {}).get(frame_id, [])
                for obj in objs:
                    if obj.class_id not in {1,2,10}:
                        continue
                    bbox = maskUtils.toBbox(obj.mask)
                    gt_boxes.append([float(v) for v in bbox])

                # Update summary
                summary["processed"] += 1
                summary["predicted_total"] += len(pred_boxes)
                summary["gt_total"] += len(gt_boxes)

                # Write predictions
                for pb, sc, pl in zip(pred_boxes, pred_scores, pred_labels):
                    x, y, w, h = pb
                    class_id = map_label_to_class(pl.split(":")[0])
                    pf.write(f"{seq_name} {frame_id} -1 {class_id} {x:.2f} {y:.2f} {w:.2f} {h:.2f} {sc:.4f}\n")
                # Write GT
                # for obj in objs:
                #     if obj.class_id not in {1,2,10}:
                #         continue
                #     bbox = maskUtils.toBbox(obj.mask)
                #     x, y, w, h = [float(v) for v in bbox]
                #     gf.write(f"{seq_name} {frame_id} {obj.track_id} {obj.class_id} {x:.2f} {y:.2f} {w:.2f} {h:.2f}\n")

# ---------------- Summary ----------------
print(f"Processed {summary['processed']} images")
print(f"Total predicted boxes: {summary['predicted_total']}")
print(f"Total GT boxes: {summary['gt_total']}")
