pushd ../../

# CUDA_VISIBLE_DEVICES=0 python train.py \
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py \
CUDA_VISIBLE_DEVICES=0 python train.py \
  --phase-name 'train' \
  --name 'voc/finetune_baseline_exp' \
  --cfg 'yolov5s.yaml' \
  --hyp 'data/hyps/hyp.finetune.yaml' \
  --data 'enot_VOC.yaml' \
  --img 640 \
  --batch 16 \
  --epochs 300 \
  --weights 'yolov5s.pt' \
  # --resume  # uncomment this if you want to resume your last train experiment

popd || exit
