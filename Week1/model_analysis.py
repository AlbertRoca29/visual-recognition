import os, time, argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm

BASE_DIR      = "/data/113-2/users/kpurkayastha/MCV/C5"
LOGS_DIR      = os.path.join(BASE_DIR, "logs")
KITTI_IMG_DIR = os.path.join(BASE_DIR, "datasets/KITTI-MOTS/training/image_02")
DEART_IMG_DIR = os.path.join(BASE_DIR, "datasets/DeART/images")
os.makedirs(LOGS_DIR, exist_ok=True)

VAL_SEQS    = [f"{i:04d}" for i in [2,6,7,8,10,13,14,16,18]]
MODEL_NAMES = ["Faster R-CNN","DETR","YOLOv10b","RT-DETR-L"]
COLORS      = ["#4C72B0","#DD8452","#55A868","#C44E52"]


def get_device():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        return torch.device("cuda:0"), n, list(range(n))
    return torch.device("cpu"), 0, []

def kitti_val_paths(n=None):
    p=[]
    for seq in VAL_SEQS:
        d=os.path.join(KITTI_IMG_DIR,seq)
        if not os.path.isdir(d): continue
        for f in sorted(os.listdir(d)):
            if f.endswith(".png"): p.append(os.path.join(d,f))
            if n and len(p)>=n: return p
    return p

def deart_paths(n=None):
    p=[]
    for root,_,files in os.walk(DEART_IMG_DIR):
        for f in sorted(files):
            if f.lower().endswith((".png",".jpg",".jpeg")):
                p.append(os.path.join(root,f))
                if n and len(p)>=n: return p
    return p

def count_params(m): return sum(p.numel() for p in m.parameters())

def warmup(fn, paths, n=5):
    for p in paths[:n]: fn(Image.open(p).convert("RGB"))
    if torch.cuda.is_available(): torch.cuda.synchronize()

def profile_model(name, fn, paths, device):
    lats=[]
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats(device)
    for p in tqdm(paths, desc=f"  {name}", leave=False):
        img=Image.open(p).convert("RGB"); t0=time.perf_counter()
        fn(img)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        lats.append(time.perf_counter()-t0)
    ms=np.mean(lats)*1000; fps=1000/ms
    mb=(torch.cuda.max_memory_allocated(device)/1e6
        if torch.cuda.is_available() else float("nan"))
    return ms, fps, mb

# ── Model builders ────────────────────────────────────────────────────────────

def build_faster_rcnn(device, gpu_ids):
    from torchvision.models.detection import (fasterrcnn_resnet50_fpn,
                                               FasterRCNN_ResNet50_FPN_Weights)
    import torchvision.transforms as T
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    npar  = count_params(model)
    if len(gpu_ids)>1: model=torch.nn.DataParallel(model,device_ids=gpu_ids)
    inner = model.module if hasattr(model,"module") else model
    model.to(device).eval(); tf=T.ToTensor()
    def infer(pil):
        with torch.no_grad(): inner([tf(pil).to(device)])
    return npar, infer

def build_detr(device, gpu_ids):
    from transformers import DetrImageProcessor, DetrForObjectDetection
    proc  = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    npar  = count_params(model)
    if len(gpu_ids)>1: model=torch.nn.DataParallel(model,device_ids=gpu_ids)
    inner = model.module if hasattr(model,"module") else model
    model.to(device).eval()
    def infer(pil):
        with torch.no_grad():
            inp=proc(images=pil,return_tensors="pt").to(device)
            out=inner(**inp)
            proc.post_process_object_detection(
                out,target_sizes=torch.tensor([pil.size[::-1]]).to(device),threshold=0.05)
    return npar, infer

def build_yolo(device, gpu_ids):
    from ultralytics import YOLO
    dev=gpu_ids if gpu_ids else "cpu"
    ckpt=os.path.join(BASE_DIR,"yolov10b.pt")
    model=YOLO(ckpt); npar=count_params(model.model)
    def infer(pil): model.predict(pil,verbose=False,device=dev)
    return npar, infer

def build_rtdetr(device, gpu_ids):
    from ultralytics import RTDETR
    dev=gpu_ids if gpu_ids else "cpu"
    model=RTDETR(os.path.join(BASE_DIR,"rtdetr-l.pt")); npar=count_params(model.model)
    def infer(pil): model.predict(pil,verbose=False,device=dev)
    return npar, infer

# ── mAP from logs ─────────────────────────────────────────────────────────────

def _parse_map(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        for line in f:
            for tok in line.split():
                try:
                    v=float(tok)
                    if 0<=v<=1: return v
                except ValueError: pass
    return None

def gather_maps():
    L=LOGS_DIR
    return {
        "Faster R-CNN": {"kitti": _parse_map(os.path.join(L,"taskef_fasterrcnn_frozen_kitti_eval.log")),
                         "deart": _parse_map(os.path.join(L,"taskef_fasterrcnn_frozen_deart_coco_full_eval.log"))},
        "DETR":         {"kitti": _parse_map(os.path.join(L,"taske_finetuned_detr_kitti_eval.log")),   "deart": None},
        "YOLOv10b":     {"kitti": _parse_map(os.path.join(L,"taske_finetuned_yolov10b_kitti_eval.log")),"deart": None},
        "RT-DETR-L":    {"kitti": _parse_map(os.path.join(L,"taskh_rtdetr_r50_frozen_kitti_eval.log")), "deart": None},
    }

# ── Charts ────────────────────────────────────────────────────────────────────

def bar_chart(vals, lbls, title, ylabel, fname):
    fig,ax=plt.subplots(figsize=(9,5))
    bars=ax.bar(lbls,vals,color=COLORS[:len(lbls)],edgecolor="white",width=0.55)
    mx=max(v for v in vals if v==v)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+mx*0.01,
                f"{v:.2f}",ha="center",va="bottom",fontsize=10,fontweight="bold")
    ax.set_title(title,fontsize=13,fontweight="bold",pad=12)
    ax.set_ylabel(ylabel,fontsize=11); ax.set_xlabel("Model",fontsize=11)
    ax.grid(axis="y",linestyle="--",alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); out=os.path.join(LOGS_DIR,fname)
    plt.savefig(out,dpi=150); plt.close(); print(f"  → {out}")

def robustness_chart(maps):
    n=len(MODEL_NAMES); x=np.arange(n); w=0.35
    kv=[maps[m]["kitti"] or 0.0 for m in MODEL_NAMES]
    dv=[maps[m]["deart"] or 0.0 for m in MODEL_NAMES]
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar(x-w/2,kv,w,label="KITTI-MOTS",color="#4C72B0",edgecolor="white")
    ax.bar(x+w/2,dv,w,label="DeART",     color="#DD8452",edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES,fontsize=10)
    ax.set_title("mAP@0.50:0.95 Robustness: KITTI-MOTS vs DeART",fontsize=12,fontweight="bold")
    ax.set_ylabel("mAP@0.50:0.95"); ax.legend()
    ax.grid(axis="y",linestyle="--",alpha=0.4)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); out=os.path.join(LOGS_DIR,"taskg_map_robustness.png")
    plt.savefig(out,dpi=150); plt.close(); print(f"  → {out}")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser=argparse.ArgumentParser(description="Task g – Model comparison")
    parser.add_argument("--num_images",type=int,default=None)
    parser.add_argument("--skip_map",action="store_true")
    args=parser.parse_args()

    device,n_gpus,gpu_ids=get_device()
    print(f"Device: {device} | GPUs: {n_gpus}")
    k_paths=kitti_val_paths(args.num_images); d_paths=deart_paths(args.num_images)
    print(f"KITTI: {len(k_paths)}  DeART: {len(d_paths)}")

    builders=[("Faster R-CNN",build_faster_rcnn),("DETR",build_detr),
              ("YOLOv10b",build_yolo),("RT-DETR-L",build_rtdetr)]
    records=[]
    for i,(name,builder) in enumerate(builders,1):
        print(f"\n[{i}/4] {name}")
        try: npar,fn=builder(device,gpu_ids)
        except Exception as e: print(f"  [SKIP] {e}"); continue
        warmup(fn,k_paths)
        k_ms,k_fps,k_mem=profile_model(name,fn,k_paths,device)
        d_ms,d_fps,_    =profile_model(name,fn,d_paths,device)
        records.append(dict(model=name,params_M=npar/1e6,
                            kitti_ms=k_ms,kitti_fps=k_fps,kitti_mem_MB=k_mem,
                            deart_ms=d_ms,deart_fps=d_fps))
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    df=pd.DataFrame(records)
    csv=os.path.join(LOGS_DIR,"taskg_model_comparison.csv")
    df.to_csv(csv,index=False,float_format="%.4f"); print(f"CSV → {csv}")
    txt=os.path.join(LOGS_DIR,"taskg_model_comparison.txt")
    with open(txt,"w") as f:
        f.write("Task g – Object Detector Comparison\n"+"="*80+"\n")
        for r in records:
            f.write(f"{r['model']:<16} Params={r['params_M']:.1f}M  "
                    f"KITTI={r['kitti_ms']:.1f}ms/{r['kitti_fps']:.1f}fps/{r['kitti_mem_MB']:.0f}MB  "
                    f"DeART={r['deart_ms']:.1f}ms\n")
    print(f"TXT → {txt}")

    lbls=[r["model"] for r in records]; print("\nCharts …")
    bar_chart([r["params_M"]    for r in records],lbls,"Parameters (M)","Params(M)","taskg_params_comparison.png")
    bar_chart([r["kitti_ms"]    for r in records],lbls,"Latency – KITTI (ms/img)","ms/img","taskg_latency_kitti.png")
    bar_chart([r["kitti_fps"]   for r in records],lbls,"Throughput – KITTI (FPS)","FPS","taskg_fps_comparison.png")
    bar_chart([r["kitti_mem_MB"]for r in records],lbls,"Peak GPU Memory – KITTI (MB)","MB","taskg_memory_kitti.png")
    if not args.skip_map:
        maps=gather_maps()
        for m in MODEL_NAMES: print(f"  {m:<15}: KITTI={maps[m]['kitti']}  DeART={maps[m]['deart']}")
        robustness_chart(maps)
    print("\nTask g complete.")

if __name__=="__main__":
    main()
