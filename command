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

python ~/eval_final.py \
  --model ~/runs/cme_yolo/yolo26n_30pct_20ep-2/weights/best.pt \
  --data ~/datasets/cme_yolo_seg_30/data.yaml \
  --original ~/synthetic_images/synthetic_images/cme_seg_20250320 \
  --outdir ~/runs/cme_yolo/yolo26n_30pct_20ep-2/eval_outputs

python ~/plot_loss.py --run_dir ~/runs/cme_yolo/yolo26n_30pct_20ep-2
python ~/plot_epoch_curves.py --run_dir ~/runs/cme_yolo/yolo26n_30pct_20ep-2



