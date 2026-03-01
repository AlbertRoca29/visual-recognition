"""
Usage:
    python scripts/analyze_models.py \
        --data_root data/kitti_mots_yolo \
        --eval_results runs/eval_results.json \
        --models YOLOv8n YOLOv8s YOLOv10n YOLOv11n \
        --max_images 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.wandb_logger import finish, init_run, log_comparison_table

MODEL_ZOO = [
    ("YOLOv8n", "yolov8n.pt"),
    ("YOLOv8s", "yolov8s.pt"),
    ("YOLOv8m", "yolov8m.pt"),
    ("YOLOv10n", "yolov10n.pt"),
    ("YOLOv10s", "yolov10s.pt"),
    ("YOLOv10m", "yolov10m.pt"),
    ("YOLOv11n", "yolo11n.pt"),
    ("YOLOv11s", "yolo11s.pt"),
    ("YOLOv11m", "yolo11m.pt"),
]

COCO_CLASSES_OF_INTEREST = [0, 2]


def add_gaussian_noise(img: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    noise = np.random.randn(*img.shape) * sigma
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_gaussian_blur(img: np.ndarray, ksize: int = 15) -> np.ndarray:
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def add_jpeg_compression(img: np.ndarray, quality: int = 20) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def shift_brightness(img: np.ndarray, delta: int = 80) -> np.ndarray:
    return np.clip(img.astype(np.int32) + delta, 0, 255).astype(np.uint8)


CORRUPTIONS = {
    "clean": lambda x: x,
    "gaussian_noise": lambda x: add_gaussian_noise(x, sigma=30),
    "blur": lambda x: add_gaussian_blur(x, ksize=15),
    "jpeg_low": lambda x: add_jpeg_compression(x, quality=15),
    "brightness": lambda x: shift_brightness(x, delta=80),
}


def profile_model(
    model: YOLO, image_paths: list[Path], device: str, n_warmup: int = 5
) -> dict:
    timings = {}

    for dev in ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]:
        # warmup
        for p in image_paths[:n_warmup]:
            model(str(p), verbose=False, device=dev, classes=COCO_CLASSES_OF_INTEREST)

        t0 = time.perf_counter()
        for p in image_paths:
            model(str(p), verbose=False, device=dev, classes=COCO_CLASSES_OF_INTEREST)
        elapsed_s = time.perf_counter() - t0

        avg_ms = elapsed_s / len(image_paths) * 1000
        timings[dev] = {"avg_ms": round(avg_ms, 2), "fps": round(1000 / avg_ms, 2)}

    return timings


def count_detections(
    model: YOLO, image_paths: list[Path], device: str, conf: float = 0.25
) -> int:
    total = 0
    for p in image_paths:
        res = model(
            str(p),
            verbose=False,
            device=device,
            conf=conf,
            classes=COCO_CLASSES_OF_INTEREST,
        )
        total += len(res[0].boxes)
    return total


def robustness_analysis(
    model: YOLO,
    image_paths: list[Path],
    device: str,
    conf: float = 0.25,
    max_imgs: int = 50,
) -> dict[str, float]:
    sample = image_paths[:max_imgs]
    results = {}

    # baseline clean count
    clean_count = 0
    for p in sample:
        img = cv2.imread(str(p))
        res = model(
            img,
            verbose=False,
            device=device,
            conf=conf,
            classes=COCO_CLASSES_OF_INTEREST,
        )
        clean_count += len(res[0].boxes)

    results["clean_det_count"] = clean_count

    for corr_name, corr_fn in CORRUPTIONS.items():
        if corr_name == "clean":
            continue
        det_count = 0
        for p in sample:
            img = cv2.imread(str(p))
            corrupted = corr_fn(img)
            res = model(
                corrupted,
                verbose=False,
                device=device,
                conf=conf,
                classes=COCO_CLASSES_OF_INTEREST,
            )
            det_count += len(res[0].boxes)

        ratio = det_count / clean_count if clean_count > 0 else 0.0
        results[f"robustness_{corr_name}"] = round(ratio, 4)
        print(f"    {corr_name:20s}: {det_count} dets  " f"(ratio={ratio:.3f})")

    return results


# ── GFLOPs via ultralytics ────────────────────────────────────────────────────


def get_flops(model: YOLO, imgsz: int = 640) -> float:
    try:
        info = model.info(verbose=False)
        # info[1] is GFLOPs in recent ultralytics versions
        return round(float(info[1]), 2)
    except Exception:
        return -1.0


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="YOLO model analysis")
    parser.add_argument("--data_root", type=Path, default=Path("data/kitti_mots_yolo"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--max_images", type=int, default=100)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--eval_results",
        type=Path,
        default=None,
        help="Optional JSON with pre-computed COCO metrics per model "
        "(output of evaluate.py). Keys: model_name → metrics dict.",
    )
    parser.add_argument(
        "--skip_robustness",
        action="store_true",
        help="Skip the corruption robustness analysis (saves time)",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--wandb_project", type=str, default="C5-Object-Detection-YOLO")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    precomputed = {}
    if args.eval_results and args.eval_results.exists():
        with open(args.eval_results) as f:
            precomputed = json.load(f)
        print(f"Loaded pre-computed metrics from {args.eval_results}")

    img_dir = args.data_root / "images" / args.split
    image_paths = sorted(img_dir.glob("*.png"))[: args.max_images]
    print(f"\nUsing {len(image_paths)} images for profiling.\n")

    zoo = MODEL_ZOO
    if args.models:
        zoo = [(n, w) for n, w in MODEL_ZOO if n in args.models]

    all_rows = []
    import wandb

    run = init_run(
        run_name="model_analysis",
        config={
            "task": "model_analysis",
            "split": args.split,
            "num_images": len(image_paths),
            "conf": args.conf,
            "models": [n for n, _ in zoo],
        },
        tags=["task_g", "analysis", "KITTI-MOTS"],
        group="task_g_analysis",
        notes="Comprehensive model comparison: speed, params, robustness",
    )

    for model_name, weights in zoo:
        print(f"  Analysing : {model_name}")

        model = YOLO(weights)
        num_params = sum(p.numel() for p in model.model.parameters())
        size_mb = Path(weights).stat().st_size / 1e6 if Path(weights).exists() else 0.0
        gflops = get_flops(model, args.imgsz)

        print(
            f"Params: {num_params/1e6:.2f} M  |  "
            f"Size: {size_mb:.1f} MB  |  "
            f"GFLOPs: {gflops}"
        )

        print("Profiling speed …")
        timings = profile_model(model, image_paths, device)
        for dev, t in timings.items():
            print(f"    [{dev:4s}] {t['avg_ms']:.1f} ms/img  |  {t['fps']:.1f} FPS")

        robustness = {}
        if not args.skip_robustness:
            print("Robustness analysis …")
            robustness = robustness_analysis(
                model, image_paths, device, args.conf, max_imgs=50
            )

        row = {
            "model": model_name,
            "params (M)": round(num_params / 1e6, 2),
            "size (MB)": round(size_mb, 2),
            "GFLOPs": gflops,
        }

        # timing columns
        for dev, t in timings.items():
            row[f"ms/img ({dev})"] = t["avg_ms"]
            row[f"FPS ({dev})"] = t["fps"]

        # optionally attach pre-computed eval metrics
        if model_name in precomputed:
            for k, v in precomputed[model_name].items():
                row[k] = v

        # robustness columns
        row.update(robustness)

        all_rows.append(row)

        # log per-model to WandB
        wandb.log(
            {
                f"params/{model_name}": row["params (M)"],
                f"size/{model_name}": row["size (MB)"],
                f"gflops/{model_name}": gflops,
                **{f"speed_{dev}/{model_name}": t["fps"] for dev, t in timings.items()},
                **{
                    f"robustness/{model_name}/{k.replace('robustness_','')}": v
                    for k, v in robustness.items()
                    if k.startswith("robustness_")
                },
            }
        )

    # ── comparison table ──────────────────────────────────────────────────────
    print("\n\nFull comparison table:")
    header = list(all_rows[0].keys()) if all_rows else []
    col_w = max(len(h) for h in header) + 2 if header else 20
    print("  " + "  ".join(h.ljust(col_w) for h in header))
    for row in all_rows:
        print("  " + "  ".join(str(row.get(h, "")).ljust(col_w) for h in header))

    log_comparison_table(all_rows, table_name="model_analysis_comparison")

    # Log a custom chart for visual comparison in WandB
    radar_keys = ["params (M)", "size (MB)", "GFLOPs"]
    for row in all_rows:
        for k in radar_keys:
            if k in row:
                wandb.log({f"radar/{k}": row[k]})

    # save results locally
    out_json = Path("runs") / "model_analysis.json"
    out_json.parent.mkdir(exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # log as WandB artifact
    art = wandb.Artifact("model_analysis", type="results")
    art.add_file(str(out_json))
    wandb.log_artifact(art)

    finish()
    print("\nAnalysis complete.\n")


if __name__ == "__main__":
    main()
