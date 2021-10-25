pushd ../../

# CUDA_VISIBLE_DEVICES=0 python train.py \
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py \
CUDA_VISIBLE_DEVICES=0 python train.py \
  --phase-name 'train' \
  --name 'vehicles/finetune_model_r448_arch_0_0_1_1_2_2_3_3_exp' \
  --cfg 'yolov5s_ss_v2.yaml' \
  --hyp 'data/hyps/hyp.finetune.yaml' \
  --data 'enot_VOC.yaml' \
  --img 448 \
  --batch 64 \
  --epochs 300 \
  --noautoanchor \
  --weights 'runs/train/coco/train_model_r448_arch_0_0_1_1_2_2_3_3_exp/weights/best.pt' \
  --architecture-indices '[0, 0, 1, 1, 2, 2, 3, 3]' \
  # --resume  # uncomment this if you want to resume your last train experiment

popd || exit
