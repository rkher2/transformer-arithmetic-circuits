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



















(yolo26) guest1@gehme-gpu:~/runs/cme_yolo$ python ~/eval_final.py \
  --model ~/runs/cme_yolo/yolo26n_30pct_20ep-2/weights/best.pt \
  --data ~/datasets/cme_yolo_seg_30/data.yaml \
  --original ~/synthetic_images/synthetic_images/cme_seg_20250320 \
  --outdir ~/runs/cme_yolo/yolo26n_30pct_20ep-2/eval_outputs

python ~/plot_loss.py --run_dir ~/runs/cme_yolo/yolo26n_30pct_20ep-2
python ~/plot_epoch_curves.py --run_dir ~/runs/cme_yolo/yolo26n_30pct_20ep-2
WARNING ⚠️ 
Inference results will accumulate in RAM unless `stream=True` is passed, which can cause out-of-memory errors for large
sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs

panoptic quality

[occulter]
  TP=3000  FP=0  FN=0
  SQ: 0.9715
  RQ:                   1.0000
  PQ:                   0.9715
/home/guest1/eval_final.py:127: UserWarning: No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.
  plt.legend()
  saved histogram -> /home/guest1/runs/cme_yolo/yolo26n_30pct_20ep-2/eval_outputs/iou_hist_occulter.png
  mean IoU: 0.9715
  median IoU: 0.9705

[cme]
  TP=2919  FP=187  FN=81
  SQ: 0.9297
  RQ:                   0.9561
  PQ:                   0.8889
  saved histogram -> /home/guest1/runs/cme_yolo/yolo26n_30pct_20ep-2/eval_outputs/iou_hist_cme.png
  mean IoU: 0.9060
  median IoU: 0.9502

 mean PQ across classes: 0.9302
PQ of CMEs: 0.8889
Traceback (most recent call last):
  File "/home/guest1/plot_loss.py", line 2, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'
Traceback (most recent call last):
  File "/home/guest1/plot_epoch_curves.py", line 49, in <module>
    main(args)
  File "/home/guest1/plot_epoch_curves.py", line 17, in main
    with open(log_csv) as f:
         ^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/home/guest1/runs/cme_yolo/yolo26n_30pct_20ep-2/epoch_metrics.csv'
(yolo26) guest1@gehme-gpu:~/runs/cme_yolo$ pip install pandas
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: pandas in /home/guest1/.local/lib/python3.14/site-packages (3.0.3)
Requirement already satisfied: numpy>=2.3.3 in /home/guest1/.local/lib/python3.14/site-packages (from pandas) (2.4.6)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/lib/python3.14/site-packages (from pandas) (2.9.0.post0)
Requirement already satisfied: six>=1.5 in /usr/lib/python3.14/site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
(yolo26) guest1@gehme-gpu:~/runs/cme_yolo$ ls
yolo26l_100pct_100ep  yolo26n_100pct_noaug  yolo26n_30pct_20ep  yolo26n_30pct_20ep-2  yolo26s_50pct_50ep
(yolo26) guest1@gehme-gpu:~/runs/cme_yolo$ cd yolo26n_30pct_20ep-2
(yolo26) guest1@gehme-gpu:~/runs/cme_yolo/yolo26n_30pct_20ep-2$ ls
args.yaml        BoxR_curve.png                   labels.jpg        MaskR_curve.png   train_batch1.jpg     train_batch3192.jpg    val_batch1_pred.jpg
BoxF1_curve.png  confusion_matrix_normalized.png  MaskF1_curve.png  results.csv       train_batch2.jpg     val_batch0_labels.jpg  val_batch2_labels.jpg
BoxP_curve.png   confusion_matrix.png             MaskP_curve.png   results.png       train_batch3190.jpg  val_batch0_pred.jpg    val_batch2_pred.jpg
BoxPR_curve.png  eval_outputs                     MaskPR_curve.png  train_batch0.jpg  train_batch3191.jpg  val_batch1_labels.jpg  weights
(yolo26) guest1@gehme-gpu:~/runs/cme_yolo/yolo26n_30pct_20ep-2$ cd epoch_metrics.csv
bash: cd: epoch_metrics.csv: No such file or directory
(yolo26) guest1@gehme-gpu:~/runs/cme_yolo/yolo26n_30pct_20ep-2$ cd eval_outputs
(yolo26) guest1@gehme-gpu:~/runs/cme_yolo/yolo26n_30pct_20ep-2/eval_outputs$ ls
iou_hist_cme.png  iou_hist_occulter.png
(yolo26) guest1@gehme-gpu:~/runs/cme_yolo/yolo26n_30pct_20ep-2/eval_outputs$ realpath iou_hist_cme.png
(yolo26) guest1@gehme-gpu:~/runs/cme_yolo/yolo26n_30pct_20ep-2/eval_outputs$ 



