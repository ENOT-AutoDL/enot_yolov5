pushd ../../

# CUDA_VISIBLE_DEVICES=0 python train.py \
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py \
CUDA_VISIBLE_DEVICES=0 python train.py \
  --phase-name 'train' \
  --name 'voc/train_model_00224466_exp' \
  --cfg 'yolov5s_ss_v2.yaml' \
  --hyp 'data/hyps/hyp.finetune.yaml' \
  --data 'enot_VOC.yaml' \
  --img 480 \
  --batch 16 \
  --epochs 300 \
  --weights 'runs/train/coco/train_model_00224466_exp/weights/best.pt' \
  --architecture-indices '[0, 0, 2, 2, 4, 4, 6, 6]' \
  # --resume  # uncomment this if you want to resume your last train experiment

popd || exit
