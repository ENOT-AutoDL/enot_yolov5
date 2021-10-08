pushd ../

python export.py \
  --weights 'weights/your_model.pt' \
  --img-size 640 \
  --batch-size 1 \
  --device cpu \
  --include onnx \
  --simplify \
  --architecture-indices '[0, 1, 2, 3, 4, 5, 6, 7]' \
  --opset 11

popd || exit
