'''
this script trains YOLO and saves statistics 
'''

import argparse
import csv
from pathlib import Path
import numpy as np
import cv2
import yaml
from ultralytics import YOLO

from seg_metrics import get_gt_mask, iou

PQ_IOU_THRESHOLD = 0.5  # textbook PQ definition, fixed
CLASS_MASK_SUFFIX = {0: "_mask_0.png", 1: "_mask_2.png"}

_folder_cache = {}


def find_original_folder(stem, source_root):
    if not _folder_cache:
        for folder in source_root.iterdir():
            if folder.is_dir():
                for img in folder.glob("*_btot.png"):
                    _folder_cache[img.stem] = folder
    return _folder_cache.get(stem)


def run_eval(model, data_yaml, original_root, imgsz, device):
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    val_dir = Path(data["path"]) / data["val"]
    source_root = Path(original_root).expanduser()

    results = model(source=str(val_dir), imgsz=imgsz, device=device, verbose=False)

    records = {0: [], 1: []}
    for r in results:
        stem = Path(r.path).stem
        original = find_original_folder(stem, source_root)
        if original is None:
            continue
        mask_dir = original / "mask"

        pred_masks_by_class = {0: [], 1: []}
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            h, w = cv2.imread(str(next(original.glob("*_btot.png")))).shape[:2]
            for mask, cls in zip(masks, classes):
                cls = int(cls)
                if cls in pred_masks_by_class:
                    pred_masks_by_class[cls].append(cv2.resize(mask, (w, h)) > 0.5)

        for cls, suffix in CLASS_MASK_SUFFIX.items():
            gt_files = list(mask_dir.glob(f"*{suffix}"))
            if not gt_files:
                continue
            gt_mask = get_gt_mask(gt_files[0])
            preds = pred_masks_by_class[cls]
            best_iou = max((iou(gt_mask, p) for p in preds), default=0.0)
            records[cls].append({"best_iou": best_iou, "num_preds": len(preds)})
    all_ious_combined = []
    pq_values = []

    for cls, recs in records.items():
        tp_ious = [r_["best_iou"] for r_ in recs if r_["best_iou"] > PQ_IOU_THRESHOLD]
        tp = len(tp_ious)
        fn = sum(1 for r_ in recs if r_["num_preds"] == 0 or r_["best_iou"] <= PQ_IOU_THRESHOLD)
        fp = sum(max(r_["num_preds"] - 1, 0) for r_ in recs)

        sq = np.mean(tp_ious) if tp_ious else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
        pq_values.append(sq * rq)

        all_ious_combined.extend([r_["best_iou"] for r_ in recs])

    mean_iou = np.mean(all_ious_combined) if all_ious_combined else 0.0
    median_iou = np.median(all_ious_combined) if all_ious_combined else 0.0
    mean_pq = np.mean(pq_values)

    return mean_iou, median_iou, mean_pq
  
def make_callback(data_yaml, original_root, eval_every, log_csv):
    def on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1
        if epoch % eval_every != 0:
            return

        weights_path = trainer.save_dir / "weights" / "last.pt"
        model = YOLO(str(weights_path))

        mean_iou, median_iou, mean_pq = run_eval(
            model, data_yaml, original_root,
            imgsz=trainer.args.imgsz, device=trainer.args.device
        )

        log_path = Path(log_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists()

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "mean_iou", "median_iou", "mean_pq"])
            writer.writerow([epoch, mean_iou, median_iou, mean_pq])

        print(f"[epoch {epoch}] mean_iou={mean_iou:.4f} median_iou={median_iou:.4f} mean_pq={mean_pq:.4f}")

    return on_fit_epoch_end


def main(args):
    data_yaml = Path(args.data).expanduser().resolve()
    original_root = Path(args.original).expanduser().resolve()

    project = Path(args.project).expanduser()
    log_csv = project / args.name / "epoch_metrics.csv"

    model = YOLO(args.model)
    model.add_callback(
        "on_fit_epoch_end",
        make_callback(data_yaml, original_root, args.eval_every, log_csv)
    )

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project),
        name=args.name,

        # --- augmentations: ALL disabled by default, enable individually via CLI ---
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        bgr=args.bgr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
        auto_augment=args.auto_augment,  # None disables it
        crop_fraction=args.crop_fraction,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # --- core training args ---
    parser.add_argument("--data", required=True, help="Path to data.yaml, e.g. ~/datasets/cme_yolo_seg_50/data.yaml")
    parser.add_argument("--original", required=True, help="Path to original source dataset (for GT masks)")
    parser.add_argument("--model", default="yolo26s-seg.pt")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--imgsz", default=512, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="~/runs/cme_yolo")
    parser.add_argument("--name", required=True, help="Run name, e.g. yolo26s_50pct")
    parser.add_argument("--eval_every", default=10, type=int)

    # --- augmentations: everything defaults to OFF/neutral ---
    parser.add_argument("--hsv_h", default=0.0, type=float)
    parser.add_argument("--hsv_s", default=0.0, type=float)
    parser.add_argument("--hsv_v", default=0.0, type=float)
    parser.add_argument("--degrees", default=0.0, type=float)
    parser.add_argument("--translate", default=0.0, type=float)
    parser.add_argument("--scale", default=0.0, type=float)
    parser.add_argument("--shear", default=0.0, type=float)
    parser.add_argument("--perspective", default=0.0, type=float)
    parser.add_argument("--flipud", default=0.0, type=float)
    parser.add_argument("--fliplr", default=0.0, type=float)
    parser.add_argument("--bgr", default=0.0, type=float)
    parser.add_argument("--mosaic", default=0.0, type=float)
    parser.add_argument("--mixup", default=0.0, type=float)
    parser.add_argument("--copy_paste", default=0.0, type=float)
    parser.add_argument("--erasing", default=0.0, type=float)
    parser.add_argument("--crop_fraction", default=1.0, type=float, help="1.0 = no crop augmentation")

    args = parser.parse_args()
    main(args)
  


