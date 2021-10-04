pushd ../

CUDA_VISIBLE_DEVICES=0 python resolution_search.py \
  --phase-name 'search' \
  --name 'resolution_search_v1_latency_3_0-0' \
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
  --latency-file './yolo_v5lss_v2_different_resolutions/'

popd
