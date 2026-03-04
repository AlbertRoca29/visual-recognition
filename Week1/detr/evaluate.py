import sys
import argparse
from collections import defaultdict
import math
import numpy as np
import cv2
from pathlib import Path
import random

from config import DB_PATH, IOU, TRAIN_IMG_DIR

VIS_DIR = Path("./eval_vis")
VIS_DIR.mkdir(exist_ok=True, parents=True)

PCT_VIS = 0.02


def draw_boxes_on_image(image_path, gt_boxes, pred_boxes, pred_labels=None, ignore_zones=None):
    img = cv2.imread(str(image_path))
    # Draw GT boxes in green
    for b in gt_boxes:
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, "GT", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    # Draw predicted boxes in red
    for i, b in enumerate(pred_boxes):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
        if pred_labels is not None:
            cv2.putText(img, pred_labels[i], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    # Draw filled black rectangles for ignore zones (class 10)
    if ignore_zones:
        for z in ignore_zones:
            zx1, zy1, zx2, zy2 = map(int, z)
            cv2.rectangle(img, (zx1, zy1), (zx2, zy2), (0,0,0), thickness=-1)
    return img

def parse_box_fields(fields):
    # fields: seq frame track_id class x y w h [score?]
    seq = fields[0]
    frame = fields[1]
    img_id = f"{seq}/{frame}"
    class_id = int(fields[3])
    x = float(fields[4])
    y = float(fields[5])
    w = float(fields[6])
    h = float(fields[7])
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    score = None
    if len(fields) >= 9:
        try:
            score = float(fields[8])
        except:
            score = None
    return img_id, class_id, (x1, y1, x2, y2), score

def load_gt(path):
    gt = defaultdict(list)
    ignore_zones = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            img_id, class_id, box, _ = parse_box_fields(parts)
            entry = {'box': box, 'class': class_id, 'used': False}
            if class_id == 10:
                ignore_zones[img_id].append(box)
            else:
                gt[img_id].append(entry)
    return gt, ignore_zones

def load_preds(path):
    preds = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            img_id, class_id, box, score = parse_box_fields(parts)
            preds[img_id].append({'box': box, 'class': class_id, 'score': score})
    return preds

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

def filter_boxes(boxes_dict, ignore_zones, ignore_class_10=False):
    def center_in_zone(box, zone):
        bx1, by1, bx2, by2 = box
        zx1, zy1, zx2, zy2 = zone
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        return zx1 <= cx <= zx2 and zy1 <= cy <= zy2

    new_dict = defaultdict(list)
    for img, boxes in boxes_dict.items():
        ignores = ignore_zones.get(img, [])
        for entry in boxes:
            b = entry['box']
            if any(center_in_zone(b, iz) for iz in ignores):
                continue
            if ignore_class_10 and entry.get('class') == 10:
                continue
            new_dict[img].append(entry)
    return new_dict

def gather_classes(gt, preds):
    classes = set()
    for img, boxes in gt.items():
        for e in boxes:
            classes.add(e['class'])
    for img, dets in preds.items():
        for d in dets:
            classes.add(d['class'])
    if 10 in classes:
        classes.remove(10)
    return sorted(classes)

def compute_ap(rec, prec):
    # Compute average precision as area under PR curve (VOC 2010+ style)
    # rec, prec must be lists/numpy arrays
    # Append sentinel values
    mrec = [0.0] + rec + [1.0]
    mpre = [0.0] + prec + [0.0]
    # precision envelope
    for i in range(len(mpre)-2, -1, -1):
        if mpre[i] < mpre[i+1]:
            mpre[i] = mpre[i+1]
    # integrate area
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            ap += (mrec[i] - mrec[i-1]) * mpre[i]
    return ap

def evaluate(gt, preds, iou_thresh=0.5):
    classes = gather_classes(gt, preds)
    per_class_ap = {}
    per_class_info = {}
    for cls in classes:
        # build GT lookup: img -> list of boxes (with used flag)
        gt_by_img = {}
        npos = 0
        for img, boxes in gt.items():
            lst = []
            for e in boxes:
                if e['class'] == cls:
                    lst.append({'box': e['box'], 'used': False})
            if lst:
                gt_by_img[img] = lst
                npos += len(lst)
        # collect predictions of this class
        detections = []
        for img, dets in preds.items():
            for d in dets:
                if d['class'] == cls:
                    detections.append({'img': img, 'box': d['box'], 'score': d.get('score')})
        # sort by score if available, else leave order (stable)
        if any(d['score'] is not None for d in detections):
            detections.sort(key=lambda x: float(x['score']) if x['score'] is not None else 0.0, reverse=True)
        # evaluate detections
        tp = []
        fp = []
        for det in detections:
            img = det['img']
            bb = det['box']
            ovmax = 0.0
            jmax = -1
            if img in gt_by_img:
                gts = gt_by_img[img]
                for j, g in enumerate(gts):
                    if g['used']:
                        continue
                    ov = iou(bb, g['box'])
                    if ov > ovmax:
                        ovmax = ov
                        jmax = j
            if ovmax >= iou_thresh and jmax >= 0:
                tp.append(1)
                fp.append(0)
                gt_by_img[img][jmax]['used'] = True
            else:
                tp.append(0)
                fp.append(1)

        # compute precision-recall
        if len(tp) == 0:
            ap = 0.0
            prec = []
            rec = []
        else:
            tp_arr = np.cumsum(tp).astype(float)
            fp_arr = np.cumsum(fp).astype(float)
            prec = tp_arr / (tp_arr + fp_arr + 1e-12)
            rec = tp_arr / (npos + 1e-12)
            ap = compute_ap(list(rec), list(prec))
        per_class_ap[cls] = ap
        per_class_info[cls] = {'AP': ap, 'npos': npos, 'ndet': len(detections)}
    # mAP
    if len(per_class_ap) == 0:
        mAP = 0.0
    else:
        mAP = sum(per_class_ap.values()) / len(per_class_ap)
    return mAP, per_class_ap, per_class_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='predictions txt file')
    parser.add_argument('--gt', required=True, help='ground-truth txt file')
    args = parser.parse_args()

    gt, ignore_zones = load_gt(args.gt)
    preds = load_preds(args.pred)

    gt = filter_boxes(gt, ignore_zones)
    preds = filter_boxes(preds, ignore_zones, ignore_class_10=True)

    common_imgs = set(gt.keys()) & set(preds.keys())
    gt = {k: v for k, v in gt.items() if k in common_imgs}
    preds = {k: v for k, v in preds.items() if k in common_imgs}

    # Evaluate at the configured IOU (keeps backward-compatible per-class output)
    mAP_cfg, per_class_ap, per_class_info = evaluate(gt, preds, iou_thresh=IOU)

    # Compute mAP@0.50 (map50)
    mAP50, _, _ = evaluate(gt, preds, iou_thresh=0.5)

    # Compute mAP@0.50:0.95 (map50-95) as the mean over IoU thresholds 0.50:0.05:0.95
    iou_thresholds = [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]
    mAPs = []
    for t in iou_thresholds:
        m_t, _, _ = evaluate(gt, preds, iou_thresh=t)
        mAPs.append(m_t)
    if len(mAPs) > 0:
        mAP50_95 = sum(mAPs) / len(mAPs)
    else:
        mAP50_95 = 0.0

    # Print summary
    print(f"mAP @ IoU={IOU:.2f}: {mAP_cfg:.6f}")
    print(f"mAP@0.50 (map50): {mAP50:.6f}")
    print(f"mAP@0.50:0.95 (map50-95): {mAP50_95:.6f}")
    print("Per-class AP:")
    for cls in sorted(per_class_ap.keys()):
        info = per_class_info[cls]
        print(f"  class {cls}: AP={per_class_ap[cls]:.6f}  GTs={info['npos']}  Dets={info['ndet']}")

    # vis
    for img_id in gt.keys():
        if random.random() > PCT_VIS:
            continue
        seq, frame = img_id.split('/')
        img_path = TRAIN_IMG_DIR / seq / f"{frame.zfill(6)}.png"

        if not img_path.exists():
            continue

        gt_boxes_img = [e['box'] for e in gt[img_id]]
        pred_boxes_img = [d['box'] for d in preds.get(img_id, [])]
        pred_labels_img = [f"{d['class']}:{d.get('score',0):.2f}" for d in preds.get(img_id, [])]

        ignore_zones_img = ignore_zones.get(img_id, [])
        vis_img = draw_boxes_on_image(img_path, gt_boxes_img, pred_boxes_img, pred_labels_img, ignore_zones=ignore_zones_img)
        out_path = VIS_DIR / f"{seq}_{frame}_eval.png"
        cv2.imwrite(str(out_path), vis_img)

if __name__ == "__main__":
    main()
