<div align="center">
<p>
<a align="left" href="https://ultralytics.com/yolov5" target="_blank">
<img width="850" src="splash.jpg"></a>
</p>
<br>
<div>
<a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv5 Citation"></a>

<br>
<p>
YOLOv5 🚀 is a family of object detection architectures and models pretrained on the COCO dataset, and represents <a href="https://ultralytics.com">Ultralytics</a>
 open-source research into future vision AI methods, incorporating lessons learned and best practices evolved over thousands of hours of research and development.
</p>

<!-- 
<a align="center" href="https://ultralytics.com/yolov5" target="_blank">
<img width="800" src="https://github.com/ultralytics/yolov5/releases/download/v1.0/banner-api.png"></a>
-->

</div>

</div>

## <div align="center">Documentation</div>

See the [YOLOv5 Docs](https://docs.ultralytics.com) for full documentation on training, testing and deployment.

See the [ENOT Docs](https://enot-autodl.rtd.enot.ai/) for full ENOT framework documentation.

## <div align="center">Getting started with the combination of ENOT and YOLOv5</div>

This project was made to simplify Neural Architecture Search of YOLOv5-like models. We currently support optimization of
two YOLOv5-like search spaces: `yolov5s` and `yolov5l`. If you want to use other `yolov5*` config - please contact the
[ENOT team](https://enot.ai/#rec321431603).

We conducted several experiments with `yolov5s`-like and `yolov5l`-like models on `MS COCO` dataset. Upon request, we
can share our pre-trained search space checkpoints. Note that we conducted all of our experiments with
<font color="red">**ReLU**</font> activation instead of `SiLU` activation. If you want to  proceed with `SiLU`
activation or another one - just replace it in `models/common.py` in `Conv` class.

Our `yolov5s`-like search space contains 8^12 (~70 billion) models, and `yolov5l`-like search space contains 8^8 models
(~17 million) models.

With ENOT framework you can reduce your model latency by sacrificing as small mean average precision as possible. To
achieve this goal, we can help you to select the best architecture and the best resolution for your problem.

To apply ENOT, you should do two additional steps.

1. The first step is named `pretrain`. During this procedure you are training all the models from your search space on 
   your own task. `Pretrain` usually takes ~5-7 times longer than single model training despite training millions of
   models.
2. The second step searches the best architecture (and, additionally, can search the best resolution) to fit your data.

To estimate gains from ENOT framework usage, you should fairly compare your baseline model and ENOT-searched model.
Baseline model is the best one you achieved by using YOLOv5 framework. You should train the model you found with ENOT
with the same setup you trained your baseline model (to exclude unfair comparison). This implies that <font color="red">
if you finetuned your baseline model from weights from another dataset (COCO, for example) - then you should first train
the model found with ENOT on this dataset, and then finetune it on your target dataset.</font>

`Pretrain` stage tips:
* You should organize `pretrain` procedure to be as close to your baseline training as possible. This means that you
  should use the same hyperparameters (augmentation, weight decay, image size, learning rate, etc). One exception is
  that `pretrain` procedure usually benefir from training with more epochs, but this would require more computation 
  resources, so we suggest keeping the same number of training epochs, too.
* `Pretrain` is a resource-intensive procedure, so you should probably use multi-GPU training to make this procedure
  faster.

`Search` stage tips:
* Copy training hyperparameters from your baseline setup. Set lr0=0.01, lrf=0.0, momentum=0.0, weight_decay=0.0,
  warmup_momentum=0.0, warmup_bias_lr=0.0.
* Use `--noautoanchor` option in search script by default. If your anchors have changed in `pretrain` procedure -
  set new anchor values in search space `.yaml` config.
* Use Adam or RAdam optimizer.
* `--max-latency-value` should be larger than the minimal model latency in the search space. You can check the minimal
  latency in the search space by calling the following code:

  ```python
  from enot.latency import min_latency, SearchSpaceLatencyContainer
  latency_container = SearchSpaceLatencyContainer.load_from_file('path_to_latency_container.pkl')
  print(min_latency(latency_container))
  ```

## <div align="center">Running ENOT optimization on MS COCO</div>

This section describes how you can reproduce our MS COCO search results.

1. Setting up your baseline
   * To start with ENOT, you need to train your baseline model. Baseline model should show your current performance (
     i.e. mean average precision at certain threshold, Precision, Recall, ...). You should also measure its execution
     time (latency), or use latency proxy such as million multiply-add operations (as used in
     [MobileNetv2](https://arxiv.org/abs/1801.04381) article).
   * Note that we conducted all of our experiments with <font color="red">**ReLU**</font> activation instead of SiLU
     activation. If your plan to proceed with another activation - change it in `models/common.py`. If you plan to use
     transfer learning from COCO dataset (by using pre-trained checkpoints) - then we cannot provide you `pretrain`
     checkpoints with activation different from ReLU.
   * To train baseline model on COCO dataset - simply go to `scripts/enot_coco/` and run `train_baseline.sh` script.
2. Pretrain procedure
   * Go to `scripts/enot_coco/` directory.
   * Launch `pretrain_search_space.sh` script to start `pretrain` procedure.
3. Search procedure
   * Select your latency kind (CPU, GPU, MMACs) and generate latency container file with `measure_latency.ipynb`
     notebook. You can also use our pre-computed latency container files in the `latency/` folder.
   * Go to `scripts/enot_coco/` directory.
   * Launch `search.sh` to search an optimal architecture for your task (you should specify `--max-latency-value` needed
     for your project).
   * Instead of searching architecture alone, you can search the optimal resolution too. Run `resolution-search.sh`
     instead of `search.sh`;
4. Training the model found with ENOT
   * Go to `scripts/enot_coco/` directory.
   * Get the last reported architecture from the search run (which is stored in list with integers), copy it and paste in
    `train_enot_model_scratch.sh` script.
   * Launch `train_enot_model_scratch.sh` script to train found architecture from scratch.

You can skip some steps. For example, we already pretrained search space based on `yolov5l` and `yolov5s` (with ReLU
activation). You should contact ENOT team if you want to use them.

## <div align="center">Transfer learning from COCO with ENOT</div>

This section describes how you can search YOLOv5-like models in COCO transfer learning setting.

1. Setting up your baseline
   * Go to `scripts/enot_pascal_voc/` directory.
   * Finetune baseline model on COCO dataset by running `finetune_baseline_from_scratch.sh` script. Here we are
     finetuning `yolov5s` model (which was pretrained on MS COCO dataset) on Pascal VOC dataset.
2. Pretrain procedure
   * Go to `scripts/enot_pascal_voc/` directory.
   * Launch `finetune_search_space.sh` script to start `pretrain` procedure. Here we are finetuning search space based on
     `yolov5s` model on Pascal VOC dataset. Our search space checkpoint is already pretrained on MS COCO dataset.
3. Search procedure
   * Select your latency kind (CPU, GPU, MMACs) and generate latency container file with `measure_latency.ipynb`
     notebook. You can also use our pre-computed latency container files in the `latency/` folder.
   * Go to `scripts/enot_pascal_voc/` directory.
   * Launch `search.sh` to search an optimal architecture for your task (you should specify `--max-latency-value` needed
     for your project).
   * Instead of searching architecture alone, you can search the optimal resolution too. Run `resolution-search.sh`
     instead of `search.sh`;
4. Training the model found with ENOT on MS COCO
   * Go to `scripts/enot_pascal_voc/` directory.
   * Get the last reported architecture from the search run (which is stored in list with integers), copy it and paste in
    `train_enot_model_scratch_coco.sh` script.
   * Launch `train_enot_model_scratch_coco.sh` script to train found architecture from scratch on MS COCO.
5. Training the model found with ENOT on your target dataset
   * Go to `scripts/enot_pascal_voc/` directory.
   * Set `--architecture-indices` as in `train_enot_model_scratch_coco.sh` script.
   * Launch `finetune_enot_model.sh` script to finetune COCO-pretrained found model on your data (Pascal VOC in this
     example).

You can skip some steps. For example, we already pretrained search space based on `yolov5l` and `yolov5s` (with ReLU
activation). You should contact ENOT team if you want to use them.

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>

[**Python>=3.7.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) installed including
[**PyTorch>=1.8**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ pip install -r requirements.txt
```

</details>

<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading models automatically from
the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
$ python detect.py --source 0  # webcam
                            file.jpg  # image 
                            file.mp4  # video
                            path/  # directory
                            path/*.jpg  # glob
                            'https://youtu.be/NUsoVlDFqZg'  # YouTube
                            'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training</summary>

Run commands below to reproduce results
on [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset (dataset auto-downloads on
first use). Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the
largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).

```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m                                40
                                         yolov5l                                24
                                         yolov5x                                16
```

</details>  

<details open>
<summary>Tutorials</summary>

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp; 🚀 RECOMMENDED
* [Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)&nbsp; ☘️
  RECOMMENDED
* [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)&nbsp; 🌟 NEW
* [Supervisely Ecosystem](https://github.com/ultralytics/yolov5/issues/2518)&nbsp; 🌟 NEW
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)&nbsp; ⭐ NEW
* [TorchScript, ONNX, CoreML Export](https://github.com/ultralytics/yolov5/issues/251) 🚀
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)&nbsp; ⭐ NEW
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)

</details>

## <div align="center">Contact</div>

For issues running YOLOv5 please visit [GitHub Issues](https://github.com/ultralytics/yolov5/issues). For business or
professional support requests please visit [https://ultralytics.com/contact](https://ultralytics.com/contact). For
issues related to ENOT framework contact [ENOT team](https://enot.ai/#rec321431603).
