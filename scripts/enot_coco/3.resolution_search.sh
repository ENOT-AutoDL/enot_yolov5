pushd ../../

CUDA_VISIBLE_DEVICES=0 python resolution_search.py \
  --phase-name 'search' \
  --name 'coco/resolution_search/latency_3_0_exp' \
  --cfg 'yolov5s_ss_v2.yaml' \
  --hyp 'data/hyps/hyp.scratch.search.yaml' \
  --data 'enot_coco.yaml' \
  --img 640 \
  --adam \
  --batch 16 \
  --epochs 20 \
  --weights 'runs/train/coco/pretrain_exp/weights/best.pt' \
  --noautoanchor \
  --max-latency-value 3.0 \
  --latency-file 'latency/yolo_v5s_ss_v2/cpu_batch_1_threads_1/'

popd || exit
