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


















(yolo26) guest1@gehme-gpu:~/runs/cme_yolo/yolo26n_30pct_20ep$ python ~/plot_loss.py --run_dir ~/runs/cme_yolo/yolo26n_30pct_20ep
Traceback (most recent call last):
  File "/home/guest1/plot_loss.py", line 2, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'
(yolo26) guest1@gehme-gpu:~/runs/cme_yolo/yolo26n_30pct_20ep$ ~/envs/yolo26/bin/python -m pip install pandas
/home/guest1/envs/yolo26/bin/python: No module named pip
(yolo26) guest1@gehme-gpu:~/runs/cme_yolo/yolo26n_30pct_20ep$ which pandas
/usr/bin/which: no pandas in (/home/guest1/envs/yolo26/bin:/home/guest1/.vscode-server/data/User/globalStorage/github.copilot-chat/debugCommand:/home/guest1/.vscode-server/data/User/globalStorage/github.copilot-chat/copilotCli:/home/guest1/.vscode-server/cli/servers/Stable-4fe60c8b1cdac1c4c174f2fb180d0d758272d713/server/bin/remote-cli:/home/guest1/.local/bin:/home/guest1/bin:/usr/local/bin:/usr/bin)
(yolo26) guest1@gehme-gpu:~/runs/cme_yolo/yolo26n_30pct_20ep$ cd ..
(yolo26) guest1@gehme-gpu:~/runs/cme_yolo$ cd ..
(yolo26) guest1@gehme-gpu:~/runs$ cd ..
(yolo26) guest1@gehme-gpu:~$ which pandas
/usr/bin/which: no pandas in (/home/guest1/envs/yolo26/bin:/home/guest1/.vscode-server/data/User/globalStorage/github.copilot-chat/debugCommand:/home/guest1/.vscode-server/data/User/globalStorage/github.copilot-chat/copilotCli:/home/guest1/.vscode-server/cli/servers/Stable-4fe60c8b1cdac1c4c174f2fb180d0d758272d713/server/bin/remote-cli:/home/guest1/.local/bin:/home/guest1/bin:/usr/local/bin:/usr/bin)
(yolo26) guest1@gehme-gpu:~$ which pip
/usr/bin/pip
(yolo26) guest1@gehme-gpu:~$ pip install pandas
Defaulting to user installation because normal site-packages is not writeable
Requirement already satisfied: pandas in ./.local/lib/python3.14/site-packages (3.0.3)
Requirement already satisfied: numpy>=2.3.3 in ./.local/lib/python3.14/site-packages (from pandas) (2.4.6)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/lib/python3.14/site-packages (from pandas) (2.9.0.post0)
Requirement already satisfied: six>=1.5 in /usr/lib/python3.14/site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
(yolo26) guest1@gehme-gpu:~$ ~/envs/yolo26/bin/python -m pip install pandas
/home/guest1/envs/yolo26/bin/python: No module named pip
(yolo26) guest1@gehme-gpu:~$ python ~/plot_loss.py --run_dir ~/runs/cme_yolo/yolo26n_30pct_20ep
Traceback (most recent call last):
  File "/home/guest1/plot_loss.py", line 2, in <module>
    import pandas as pd
ModuleNotFoundError: No module named 'pandas'
(yolo26) guest1@gehme-gpu:~$ 
