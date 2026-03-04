import cv2
from pathlib import Path
from pycocotools import mask as maskUtils

# ---------------- Visualization ----------------
def draw_boxes(img, boxes, labels=None, color=(255,0,0), width=2):
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, width)
        if labels is not None:
            cv2.putText(img, labels[i], (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

# ---------------- Label Mapping ----------------
def map_label_to_class(label_name: str):
    l = label_name.lower()
    if "car" in l or "truck" in l or "van" in l:
        return 1
    if "person" in l or "pedestrian" in l or "people" in l:
        return 2
    return None

# ---------------- IOU ----------------
def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    areaA = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    areaB = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union

# ---------------- Mask Utilities ----------------
class SegmentedObject:
    def __init__(self, mask, class_id, track_id):
        self.mask = mask
        self.class_id = class_id
        self.track_id = track_id

def load_annotations(path):
    objects_per_frame = {}
    track_ids_per_frame = {}
    combined_mask_per_frame = {}

    lines = Path(path).read_text().splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        frame, track_id, class_id, width, height, counts = line.split(" ")
        frame = int(frame)
        track_id = int(track_id)
        class_id = int(class_id)
        width = int(width)
        height = int(height)
        mask = {"size": [width, height], "counts": counts.encode("utf-8")}

        objects_per_frame.setdefault(frame, [])
        track_ids_per_frame.setdefault(frame, set())

        if track_id in track_ids_per_frame[frame]:
            raise AssertionError(f"Duplicate track id {track_id} in frame {frame}")
        track_ids_per_frame[frame].add(track_id)

        if class_id not in {1, 2, 10}:
            raise AssertionError(f"Unknown object class {class_id}")

        if frame not in combined_mask_per_frame:
            combined_mask_per_frame[frame] = mask
        else:
            merged_mask = maskUtils.merge([combined_mask_per_frame[frame], mask], intersect=False)
            if maskUtils.area(maskUtils.merge([combined_mask_per_frame[frame], mask], intersect=True)) > 0:
                raise AssertionError(f"Overlapping masks in frame {frame}")
            combined_mask_per_frame[frame] = merged_mask

        objects_per_frame[frame].append(SegmentedObject(mask, class_id, track_id))

    return objects_per_frame
