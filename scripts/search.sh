pushd ../

CUDA_VISIBLE_DEVICES=0 python train.py \
  --phase-name 'search' \
  --name 'search_v1_latency_3_0-0' \
  --cfg 'yolov5sss_v1.yaml' \
  --hyp 'data/hyps/hyp.scratch.search.yaml' \
  --data 'enot_coco.yaml' \
  --img 640 \
  --adam \
  --batch 16 \
  --epochs 20 \
  --weights 'best_pretrain_checkpoint.pt' \
  --noautoanchor \
  --max-latency-value 3.0 \
  --latency-file 'yolov5lss_v2_latency.pkl' \
  # --resume  # uncomment this if you want to resume your last search experiment

popd
