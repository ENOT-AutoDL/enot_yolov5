pushd ../../

# CUDA_VISIBLE_DEVICES=0 python train.py \
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py \
CUDA_VISIBLE_DEVICES=0 python train.py \
  --phase-name 'train' \
  --name 'coco/train_model_00224466_exp' \
  --cfg 'yolov5s_ss_v2.yaml' \
  --hyp 'data/hyps/hyp.scratch.yaml' \
  --data 'enot_coco128.yaml' \
  --img 480 \
  --batch 16 \
  --epochs 300 \
  --weights '' \
  --architecture-indices '[0, 0, 2, 2, 4, 4, 6, 6]' \
  # --resume  # uncomment this if you want to resume your last train experiment

popd || exit
