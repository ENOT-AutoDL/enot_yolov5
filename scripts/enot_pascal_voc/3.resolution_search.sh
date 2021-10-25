pushd ../../

CUDA_VISIBLE_DEVICES=0 python resolution_search.py \
  --phase-name 'search' \
  --name 'voc/resolution_search/latency_7400_exp' \
  --cfg 'yolov5s_ss_v2.yaml' \
  --hyp 'data/hyps/hyp.scratch.search.yaml' \
  --data 'enot_VOC.yaml' \
  --img 640 \
  --adam \
  --batch 64 \
  --epochs 20 \
  --noautoanchor \
  --weights 'runs/train/voc/pretrain_finetune_from_coco_exp/weights/best.pt' \
  --max-latency-value 7400 \
  --latency-file 'latency/yolov5s_ss_v1_mmac/'

popd || exit
