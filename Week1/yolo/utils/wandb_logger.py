"""
Weights & Biases logging utilities for YOLO C5 project
=======================================================
Centralised helper so every script uses the same WandB setup.
"""

import os
import time
from pathlib import Path
from typing import Any

import wandb


# ── Project defaults (override via env vars or function args) ─────────────────
DEFAULT_PROJECT = os.getenv("WANDB_PROJECT", "C5-Object-Detection-YOLO")
DEFAULT_ENTITY  = os.getenv("WANDB_ENTITY",  None)   # your WandB username/team


# ── Init ──────────────────────────────────────────────────────────────────────

def init_run(
    run_name: str,
    config: dict,
    tags: list[str] | None = None,
    group: str | None = None,
    project: str = DEFAULT_PROJECT,
    entity: str | None = DEFAULT_ENTITY,
    notes: str = "",
) -> wandb.sdk.wandb_run.Run:
    """
    Initialise a WandB run and return it.

    Args:
        run_name : Human-readable name for this run (e.g. 'yolov8n_inference')
        config   : Hyperparameters / settings dict to log
        tags     : Optional list of tags (e.g. ['yolov8', 'inference', 'pretrained'])
        group    : WandB group to cluster related runs (e.g. 'task_c_inference')
        project  : WandB project name
        entity   : WandB entity (user or team). None → WandB default.
    """
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
        tags=tags or [],
        group=group,
        notes=notes,
        reinit=True,
    )
    print(f"\n🔗  WandB run : {run.url}\n")
    return run


# ── Metric helpers ────────────────────────────────────────────────────────────

def log_coco_metrics(metrics: dict, step: int | None = None, prefix: str = ""):
    """
    Log standard COCO AP metrics.

    Expected keys (from pycocotools / torchmetrics):
      map, map_50, map_75, map_small, map_medium, map_large,
      mar_1, mar_10, mar_100
    """
    log_dict = {}
    for k, v in metrics.items():
        key = f"{prefix}/{k}" if prefix else k
        log_dict[key] = float(v) if hasattr(v, "item") else v

    if step is not None:
        wandb.log(log_dict, step=step)
    else:
        wandb.log(log_dict)


def log_model_profile(
    model_name: str,
    num_params: int,
    model_size_mb: float,
    inference_time_ms: float,
    fps: float,
    dataset: str = "KITTI-MOTS",
):
    """Log model profiling data as a WandB table row (call once per model)."""
    table = wandb.Table(
        columns=["model", "dataset", "params (M)", "size (MB)",
                 "inference (ms/img)", "FPS"],
        data=[[
            model_name,
            dataset,
            round(num_params / 1e6, 2),
            round(model_size_mb, 2),
            round(inference_time_ms, 2),
            round(fps, 2),
        ]],
    )
    wandb.log({f"profile/{model_name}": table})


def log_prediction_images(
    images_paths: list[Path],
    results_list,                  # list of ultralytics Result objects
    max_images: int = 20,
    prefix: str = "predictions",
):
    """
    Log inference visualisations to WandB.

    Args:
        images_paths : Paths of the original images
        results_list : Ultralytics Results (one per image)
        max_images   : Cap to avoid huge uploads
        prefix       : WandB key prefix
    """
    wandb_images = []
    for img_path, result in zip(images_paths[:max_images], results_list[:max_images]):
        # result.plot() returns an RGB numpy array
        vis = result.plot()
        wandb_images.append(
            wandb.Image(vis, caption=Path(img_path).name)
        )
    wandb.log({prefix: wandb_images})


def log_comparison_table(rows: list[dict], table_name: str = "model_comparison"):
    """
    Log a multi-model comparison table.

    rows: list of dicts, each dict is one model's results.
    Example:
        [
          {"model": "yolov8n", "mAP50": 0.45, "mAP50-95": 0.30, "fps": 120},
          {"model": "yolov10s", "mAP50": 0.48, ...},
        ]
    """
    if not rows:
        return
    columns = list(rows[0].keys())
    data    = [[r.get(c, "") for c in columns] for r in rows]
    table   = wandb.Table(columns=columns, data=data)
    wandb.log({table_name: table})


# ── Timing context manager ────────────────────────────────────────────────────

class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name    = name
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = (time.perf_counter() - self._start) * 1000  # ms


# ── Finish ────────────────────────────────────────────────────────────────────

def finish():
    """Finalise the WandB run."""
    wandb.finish()
    print("\n✅  WandB run finished.")
