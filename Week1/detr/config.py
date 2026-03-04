from pathlib import Path

# ---------------- Dataset ----------------
DB_PATH = Path.cwd().parents[3] / "mcv" / "datasets" / "C5" / "KITTI-MOTS"
INSTANCES_DIR = DB_PATH / "instances_txt"
TRAIN_IMG_DIR = DB_PATH / "training" / "image_02"

# ---------------- Evaluation ----------------
IOU = 0.5
