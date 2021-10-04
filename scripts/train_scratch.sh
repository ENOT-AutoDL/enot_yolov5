pushd ../

# CUDA_VISIBLE_DEVICES=0 python train.py \
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py \
CUDA_VISIBLE_DEVICES=0 python train.py \
  --phase-name 'train' \
  --cfg 'yolov5l6ss_v2.yaml' \
  --hyp 'data/hyps/hyp.scratch-p6.yaml' \
  --data 'enot_coco128.yaml' \
  --img 1280 \
  --batch 4 \
  --epochs 10 \
  --weights '' \
  --noautoanchor \
  --save_period 25 \
  --architecture-indices '[7, 7, 2, 3, 0, 1, 2, 3, 0, 6, 1]' \
  # --resume  # uncomment this if you want to resume your last train experiment


popd
