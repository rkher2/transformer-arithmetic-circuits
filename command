python ~/train_with_tracking.py \
  --data ~/datasets/cme_yolo_seg_50/data.yaml \
  --original ~/synthetic_images/synthetic_images/cme_seg_20250320 \
  --model yolo26l-seg.pt \
  --name yolo26l_50pct_100ep \
  --epochs 100 \
  --imgsz 512 \
  --batch 16 \
  --device 0 \
  --eval_every 2
