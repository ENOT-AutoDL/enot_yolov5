pushd ../../

CUDA_VISIBLE_DEVICES=0 python train.py \
  --phase-name 'search' \
  --name 'voc/search/latency_3_0_exp' \
  --cfg 'yolov5s_ss_v2.yaml' \
  --hyp 'data/hyps/hyp.scratch.search.yaml' \
  --data 'enot_VOC.yaml' \
  --img 640 \
  --adam \
  --batch 16 \
  --epochs 20 \
  --weights 'runs/train/voc/pretrain_exp/weights/best.pt' \
  --noautoanchor \
  --max-latency-value 3.0 \
  --latency-file 'latency/yolo_v5s_ss_v2/cpu_batch_1_threads_1/r640.pkl'

popd || exit
