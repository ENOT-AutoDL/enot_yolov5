pushd ../../

# CUDA_VISIBLE_DEVICES=0 python train.py \
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 train.py \
CUDA_VISIBLE_DEVICES=0 python train.py \
  --phase-name 'pretrain' \
  --name 'voc/pretrain_finetune_from_coco_exp' \
  --cfg 'yolov5s_ss_v2.yaml' \
  --hyp 'data/hyps/hyp.finetune.yaml' \
  --data 'enot_VOC.yaml' \
  --img 640 \
  --batch 64 \
  --epochs 300 \
  --noautoanchor \
  --weights 'weights/yolov5s_ss_v1_relu_checkpoint.pt' \
  # --resume  # uncomment this if you want to resume your last pretrain experiment

popd || exit
