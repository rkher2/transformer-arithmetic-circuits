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



u:~/runs/cme_yolo$ rm -r yolo26n_30pct_20ep
guest1@gehme-gpu:~/runs/cme_yolo$ ls
yolo26l_100pct_100ep-2  yolo26n_100pct_50ep_noise_flip_rot  yolo26n_30pct_20ep-2
yolo26n_100pct_50ep     yolo26n_100pct_noaug                yolo26n_30pct_20ep_noise_flip_rot
guest1@gehme-gpu:~/runs/cme_yolo$ rm -r yolo26n_100pct_noaug
guest1@gehme-gpu:~/runs/cme_yolo$ rm -r yolo26n_100pct_50ep
guest1@gehme-gpu:~/runs/cme_yolo$ rm -r yolo26l_100pct_100ep-2
guest1@gehme-gpu:~/runs/cme_yolo$ ls
yolo26n_100pct_50ep_noise_flip_rot  yolo26n_30pct_20ep-2  yolo26n_30pct_20ep_noise_flip_rot
guest1@gehme-gpu:~/runs/cme_yolo$ cd yolo26n_30pct_20ep-2
guest1@gehme-gpu:~/runs/cme_yolo/yolo26n_30pct_20ep-2$ ls
args.yaml        confusion_matrix_normalized.png  MaskP_curve.png   train_batch0.jpg     train_batch3192.jpg    val_batch1_pred.jpg
BoxF1_curve.png  confusion_matrix.png             MaskPR_curve.png  train_batch1.jpg     train_losses.png       val_batch2_labels.jpg
BoxP_curve.png   eval_outputs                     MaskR_curve.png   train_batch2.jpg     val_batch0_labels.jpg  val_batch2_pred.jpg
BoxPR_curve.png  labels.jpg                       results.csv       train_batch3190.jpg  val_batch0_pred.jpg    val_losses.png
BoxR_curve.png   MaskF1_curve.png                 results.png       train_batch3191.jpg  val_batch1_labels.jpg  weights
guest1@gehme-gpu:~/runs/cme_yolo/yolo26n_30pct_20ep-2$ cd ..
guest1@gehme-gpu:~/runs/cme_yolo$ ls
yolo26n_100pct_50ep_noise_flip_rot  yolo26n_30pct_20ep-2  yolo26n_30pct_20ep_noise_flip_rot
guest1@gehme-gpu:~/runs/cme_yolo$ cd ..
guest1@gehme-gpu:~/runs$ cd ..
guest1@gehme-gpu:~$ nvidia-smi
Mon Jul 13 16:33:50 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.159.03             Driver Version: 580.159.03     CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        Off |   00000000:0F:00.0 Off |                  N/A |
| 64%   73C    P2            247W /  350W |    6616MiB /  24576MiB |     48%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 3080 Ti     Off |   00000000:10:00.0 Off |                  N/A |
|  0%   45C    P8             19W /  350W |      15MiB /  12288MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2147      G   /usr/bin/gnome-shell                    117MiB |
|    0   N/A  N/A            3846      G   /usr/bin/Xwayland                         6MiB |
|    0   N/A  N/A         1805052      C   python                                 6450MiB |
|    1   N/A  N/A            2147      G   /usr/bin/gnome-shell                      3MiB |
