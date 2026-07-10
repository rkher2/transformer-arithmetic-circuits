# eval_final.py

import argparse
from pathlib import Path
import numpy as np
import cv2
import yaml
import matplotlib.pyplot as plt
from ultralytics import YOLO
from seg_metrics import get_gt_mask, iou

# dicts defining classes
CLASS_MASK_SUFFIX = {0: "_mask_0.png", 1: "_mask_2.png"}
CLASS_NAMES = {0: "occulter", 1: "cme"}

# definition from paper about PQ
PQ_IOU_THRESHOLD = 0.5  

# find the original folder containing an image (e.g. ~/train/0 or ~/train/521)
def find_original_folder(image_stem, source_root, cache={}):
    if not cache:
        for folder in source_root.iterdir():
            if folder.is_dir():
                for img in folder.glob("*_btot.png"):
                    cache[img.stem] = folder
    return cache.get(image_stem)

def main(args):
    # load trained model + validation images
    model = YOLO(args.model)
    # read data.yaml config file to reach data
    with open(args.data) as f:
        data = yaml.safe_load(f)
    val_dir = Path(data["path"]) / data["val"]
    source_root = Path(args.original)

    # run inference to get results
    results = model(source=str(val_dir), imgsz=args.imgsz, device=args.device, verbose=False)

    records = {0: [], 1: []}
    unexpected_multi_gt = []

    # loop thru every image 
    for r in results:
        stem = Path(r.path).stem
        # recover its original folder
        original = find_original_folder(stem, source_root)
        if original is None:
            print("Missing original for:", stem)
            continue

        # find the mask files corresponding to the image 
        mask_dir = original / "mask"
        pred_masks_by_class = {0: [], 1: []}
        if r.masks is not None:
            # get masks, their corresponding classes
            masks = r.masks.data.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()
            h, w = cv2.imread(str(next(original.glob("*_btot.png")))).shape[:2]
            for mask, cls in zip(masks, classes):
                cls = int(cls)
                if cls not in pred_masks_by_class:
                    continue
                # convert values to True/False 
                resized = cv2.resize(mask, (w, h)) > 0.5
                pred_masks_by_class[cls].append(resized)

        # process occulter first then CME
        for cls, suffix in CLASS_MASK_SUFFIX.items():
            gt_files = list(mask_dir.glob(f"*{suffix}"))
            if not gt_files:
                continue

            # load binary mask for this object
            gt_mask = get_gt_mask(gt_files[0])

            # count multiple disconnected regions if part of same object
            num_components, _ = cv2.connectedComponents(gt_mask.astype(np.uint8))
            if num_components - 1 > 1:
                unexpected_multi_gt.append((stem, CLASS_NAMES[cls], num_components - 1))

            preds = pred_masks_by_class[cls]
            # pick the highest-matching IoU. for example, if the model predicts 3 CME, pick the one with highest IoU to evaluate overlap with
            # the ground-truth CME (with PQ it will get penalized for this by the other term anyway)
            best_iou = max((iou(gt_mask, p) for p in preds), default=0.0)
            # save records (the number of predicted masks and the best IoU) 
            records[cls].append({"best_iou": best_iou, "num_preds": len(preds)})
    
    print(f"panoptic quality")
    pq_per_class = {}
    pq_values = []
    
    for cls, name in CLASS_NAMES.items():
        recs = records[cls]

        # true positives
        tp_ious = [r_["best_iou"] for r_ in recs if r_["best_iou"] > PQ_IOU_THRESHOLD]
        tp = len(tp_ious)

        # false negatives
        fn = sum(1 for r_ in recs if r_["num_preds"] == 0 or r_["best_iou"] <= PQ_IOU_THRESHOLD)

        # false positives
        fp = sum(max(r_["num_preds"] - 1, 0) for r_ in recs)

        # calculate PQ 
        sq = np.mean(tp_ious) if tp_ious else 0.0
        rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) > 0 else 0.0
        pq = sq * rq
        pq_per_class[name] = pq

        pq_values.append(pq)

        print(f"\n[{name}]")
        print(f"  TP={tp}  FP={fp}  FN={fn}")
        print(f"  SQ: {sq:.4f}")
        print(f"  RQ:                   {rq:.4f}")
        print(f"  PQ:                   {pq:.4f}")

        all_best_ious = [r_["best_iou"] for r_ in recs]
        plt.figure()
        plt.hist(all_best_ious, bins=20, range=(0, 1))
        plt.axvline(PQ_IOU_THRESHOLD, color="red", linestyle="--")
        plt.title(f"best-match IoU per image: {name} (n={len(all_best_ious)})")
        plt.xlabel("IoU")
        plt.ylabel("count")
        plt.legend()
        out_path = Path(args.outdir) / f"iou_hist_{name}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        print(f"  saved histogram -> {out_path}")
        print(f"  mean IoU: {np.mean(all_best_ious):.4f}")
        print(f"  median IoU: {np.median(all_best_ious):.4f}")

    print(f"\n mean PQ across classes: {np.mean(pq_values):.4f}")
    print(f"PQ of CMEs: {pq_per_class['cme']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", default=512, type=int)
    parser.add_argument("--outdir", default="~/eval_outputs")
    args = parser.parse_args()
    args.outdir = str(Path(args.outdir).expanduser())
    main(args)
