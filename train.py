# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from enot.latency import SearchSpaceLatencyContainer
from enot.latency import current_latency
from enot.latency import max_latency
from enot.models import SearchSpaceModel
from enot.models import SearchVariantsContainer
from enot.optimize import build_enot_optimizer
from enot.utils.batch_norm import tune_bn_stats
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim import SGD
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

# Expanded imports.
import val  # for end-of-epoch mAP
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.bn_dataloader_wrapper import BatchNormTuneDataLoaderWrapper
from utils.callbacks import Callbacks
from utils.checkpoint import enable_checkpoint
from utils.datasets import create_dataloader
from utils.downloads import attempt_download
from utils.general import check_dataset
from utils.general import check_file
from utils.general import check_img_size
from utils.general import check_requirements
from utils.general import colorstr
from utils.general import get_latest_run
from utils.general import increment_path
from utils.general import init_seeds
from utils.general import labels_to_class_weights
from utils.general import labels_to_image_weights
from utils.general import methods
from utils.general import one_cycle
from utils.general import print_mutation
from utils.general import set_logging
from utils.general import strip_optimizer
from utils.loggers import Loggers
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.plots import plot_labels
from utils.torch_utils import ModelEMA
from utils.torch_utils import de_parallel
from utils.torch_utils import intersect_dicts
from utils.torch_utils import select_device
from utils.torch_utils import torch_distributed_zero_first

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def set_checkpoint(
        model_to_optimize: nn.Module,
        checkpoint_model: nn.Module,
        checkpoint_file: str,
        exclude: List[str],
) -> None:
    dict_to_load = intersect_dicts(
        checkpoint_model.float().state_dict(),  # Checkpoint state_dict as FP32.
        model_to_optimize.state_dict(),
        exclude=exclude,
    )  # Intersect checkpoints.
    model_to_optimize.load_state_dict(dict_to_load, strict=False)  # load
    LOGGER.info(
        f'Transferred {len(dict_to_load)}/{len(model_to_optimize.state_dict())} '
        f'items from {checkpoint_file}'
    )  # Report intersection statistics.
    if len(dict_to_load) == 0:
        raise RuntimeError('Unable to transfer anything from the checkpoint')


def create_and_prepare_model(
        weights: str,
        device,
        phase_name: str,
        cfg,
        opt,
        nc: int,
        hyp: Dict[str, Any],
        resume: bool,
        is_enot_phase: bool,
) -> Tuple[nn.Module, nn.Module, Optional[SearchSpaceModel], Dict[str, Any]]:

    pretrained: bool = weights.endswith('.pt')

    checkpoint: Optional[Dict[str, Any]] = None
    checkpoint_config = cfg
    checkpoint_model: Optional[nn.Module] = None
    checkpoint_with_ss: bool = False
    checkpoint_with_model_from_ss: bool = False
    checkpoint_exclude: List[str] = []

    if pretrained:

        with torch_distributed_zero_first(RANK):
            weights = attempt_download(weights)  # Download if not found locally.

        checkpoint = torch.load(weights, map_location=device)  # Load checkpoint.
        if 'phase_name' not in checkpoint:  # When using checkpoints from ultralytics.
            checkpoint['phase_name'] = 'train'

        if checkpoint['phase_name'] == 'search':
            raise RuntimeError('You should not use search phase checkpoints')

        if 'ema' in checkpoint and checkpoint['ema'] is not None and not resume:
            raise RuntimeError(
                'To prevent hard-to-find bugs, we allow using only finalized checkpoints '
                'for fine-tuning from pretrained weights\n'
                'Finalized checkpoints can be obtained by calling "strip_optimizer" function '
                '(BE CAREFUL, IT OVERWRITES YOUR CHECKPOINT FILE BY DEFAULT). This function '
                'is automatically applied to your checkpoints at the end of training).'
            )

        checkpoint_model = checkpoint['model']
        checkpoint_with_ss = isinstance(checkpoint_model, SearchSpaceModel)
        checkpoint_with_model_from_ss = any(
            isinstance(layer, SearchVariantsContainer)
            for layer in checkpoint_model.modules()
        ) and not checkpoint_with_ss
        checkpoint_config = checkpoint_model.original_model.yaml if checkpoint_with_ss else checkpoint_model.yaml
        checkpoint_exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys

    model_config = cfg or checkpoint_config
    model = Model(model_config, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    model_to_optimize: Union[nn.Module, SearchSpaceModel] = model
    search_space: Optional[SearchSpaceModel] = None
    if any(isinstance(layer, SearchVariantsContainer) for layer in model.modules()):
        model_to_optimize = search_space = SearchSpaceModel(model).to(device)

    # Verify that search space is provided for ENOT phases.
    if is_enot_phase and search_space is None:
        raise ValueError(
            f'"{model_config}" model yaml file does not contain search blocks, '
            f'but this is required for phase "{phase_name}"'
        )

    # Check whether we should extract single architecture from the search space or not.
    has_architecture_indices = opt.architecture_indices != '[]'
    extract_single_architecture = False
    if is_enot_phase or search_space is None:
        if has_architecture_indices:
            raise ValueError(
                f'--architecture-indices commandline argument should only be provided '
                f'when training single model from search space.\n'
                f'--architecture-indices should not be set for phase "{phase_name}" '
                f'and "{model_config}" model yaml file.'
            )
    else:
        if has_architecture_indices:
            extract_single_architecture = True
        else:
            raise ValueError(
                f'Training or tuning models from a search space requires setting '
                f'--architecture-indices commandline argument.\n'
                f'No architecture indices were provided for phase "{phase_name}" '
                f'and "{model_config}" model yaml file.'
            )

    # Check that we can restore checkpoint weights in the first place.
    if pretrained:
        if is_enot_phase and not checkpoint_with_ss:
            # We can only apply weights from SearchSpaceModel in ENOT phases (pretrain and search).
            raise ValueError(
                f'ENOT phase "{phase_name}" requires checkpoint with SearchSpaceModel, '
                f'but checkpoint "{weights}" has a model with type {type(checkpoint_model)}.'
            )
        if not is_enot_phase:
            if (search_space is not None) and (not checkpoint_with_ss) and (not checkpoint_with_model_from_ss):
                # New model has search variant containers, but checkpoint contains
                # regular model without search variants.
                raise ValueError(
                    f'"{model_config}" model yaml file contains search variants, '
                    f'but the model from checkpoint "{weights}" is neither a search '
                    f'space instance nor an extracted model from a search space'
                )
            if (search_space is None) and (checkpoint_with_ss or checkpoint_with_model_from_ss):
                # New model has no search variant containers, but checkpoint contains
                # either a search space or a model from the search space.
                raise ValueError(
                    f'"{model_config}" model yaml file contains no search variants, '
                    f'but the model from checkpoint "{weights}" is either a search '
                    f'space instance or an extracted model from a search space'
                )

    # For almost all cases, we can load model from our checkpoint to newly created model.
    # The only exception is loading weights from a single model from the search space.
    if pretrained and not checkpoint_with_model_from_ss:
        set_checkpoint(model_to_optimize, checkpoint_model, weights, checkpoint_exclude)

    if extract_single_architecture:
        extracted_model = search_space.get_network_by_indexes(json.loads(opt.architecture_indices))
        model = model_to_optimize = extracted_model
        search_space = None

    # Loading weights from a single model from the search space.
    if checkpoint_with_model_from_ss:
        set_checkpoint(model_to_optimize, checkpoint_model, weights, checkpoint_exclude)

    # Optionally prune search space to remove rare or unused operations.
    # from utils.prune_search_space_variants import prune_search_space_variants
    # prune_indices = [[ 0,  2,  3,  7],
    #    [ 2,  4,  7,  9],
    #    [ 1,  7,  8, 10],
    #    [ 0,  1,  6,  9],
    #    [ 6,  7,  8,  9],
    #    [ 5,  6,  9, 10],
    #    [ 0,  3,  6,  8],
    #    [ 3,  5,  6,  9]]
    # prune_search_space_variants(search_space, prune_indices=prune_indices)

    return model_to_optimize, model, search_space, checkpoint


def train(hyp,  # path/to/hyp.yaml or hyp dictionary
          opt,
          device,
          callbacks=Callbacks()
          ):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    # This function prepares data from the dataloader to pass input images to the model during batch norm tuning.
    def preprocess_data(x: Tuple[torch.Tensor]) -> Tuple[Tuple[torch.Tensor], Dict]:
        return (x[0].to(device).float() / 255.0, ), {}

    phase_name = opt.phase_name
    is_pretrain = phase_name == 'pretrain'
    is_search = phase_name == 'search'
    is_enot_phase = is_pretrain or is_search

    # Directories
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    data_dict = None

    # Loggers
    if RANK in [-1, 0]:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            if resume:
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'
    init_seeds(1 + RANK)
    with torch_distributed_zero_first(RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
    train_path, val_path = data_dict[phase_name], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    is_coco = data.endswith('coco.yaml') and nc == 80  # COCO dataset

    # Model
    pretrained = weights.endswith('.pt')
    model_to_optimize, model, search_space, ckpt = create_and_prepare_model(
        weights, device, phase_name, cfg, opt, nc, hyp, resume, is_enot_phase,
    )

    if not is_search and opt.grad_checkpoint:
        LOGGER.info('Enabling gradient checkpoint for better GPU memory utilization')
        enable_checkpoint(model_to_optimize)

    # Freeze
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False

    # Optimizer
    if is_search:  # We don't need to accumulate gradients during search procedure.
        nbs = batch_size  # nominal batch size
        accumulate = 1  # accumulate loss before optimizing
    else:
        nbs = 64  # nominal batch size
        accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    if not is_search:
        # Select parameters to optimize and later choose only params from this set.
        params_to_optimize = set(
            search_space.model_parameters()
            if is_pretrain
            else model_to_optimize.parameters()
        )
        for v in model_to_optimize.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter) and v.bias in params_to_optimize:  # bias
                g2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter) and v.weight in params_to_optimize:  # weight (with decay)
                g1.append(v.weight)
    else:
        # Use architecture parameters during search procedure.
        g0 = list(search_space.architecture_parameters())

    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # These param groups are redundant for search procedure.
    if not is_search:
        optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
        optimizer.add_param_group({'params': g2})  # add g2 (biases)
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2

    # Scheduler
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA (should not be applied during search procedure)
    ema = ModelEMA(model_to_optimize) if (RANK in [-1, 0] and not is_search) else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        if phase_name == ckpt['phase_name']:

            LOGGER.info(f'Restoring train state for the same phase ({phase_name})')

            # Optimizer
            if ckpt['optimizer'] is not None:
                optimizer.load_state_dict(ckpt['optimizer'])
                best_fitness = ckpt['best_fitness']

            # EMA
            if ema and ckpt.get('ema'):
                ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
                ema.updates = ckpt['updates']

            # Epochs
            start_epoch = ckpt['epoch'] + 1
            if resume:
                assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
            if epochs < start_epoch:
                LOGGER.info(
                    f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs."
                )
                epochs += ckpt['epoch']  # finetune additional epochs

            del ckpt

        else:

            LOGGER.info(f'Dropping training state because phase has changed ({ckpt["phase_name"]} -> {phase_name})')

    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        # DP is not supported in ENOT, raise.
        raise RuntimeError('DP is not supported in ENOT framework, use torch.distributed.run instead.\n'
                           'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')

    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        raise RuntimeError('SyncBatchNorm is supported, but you should contact ENOT developers to use it.')

    # Trainloader
    augment = not is_search  # We should not augment search data during search procedure.
    pad = 0.5 if is_search else 0.0  # Use the same padding as in validation data loader.
    use_rect = opt.rect or is_search  # Use rectangular images as in validation data loader.
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                              hyp=hyp, augment=augment, cache=opt.cache, rect=use_rect, rank=RANK,
                                              workers=workers, image_weights=opt.image_weights, quad=opt.quad,
                                              pad=pad, prefix=colorstr(f'{phase_name}: '))
    bn_tune_loader = train_loader  # Use pretrain data loader for batch norm tuning.
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in [-1, 0]:
        val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                       hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
                                       workers=workers, pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            labels = np.concatenate(dataset.labels, 0)
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        callbacks.on_pretrain_routine_end()

    # Not necessary with ENOT framework, only keep for train phase.
    # DDP mode
    if cuda and RANK != -1 and not is_enot_phase:
        model_to_optimize = DDP(model_to_optimize, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Try to load latency container for latency-aware search.
    # You can find latency containers in yolov5/latency folder.
    if is_search:
        latency_file = opt.latency_file
        if latency_file == '' and opt.max_latency_value != 0.0:
            raise ValueError('Latency file is not provided when max-latency-value is positive')
        if latency_file != '':
            latency_container = SearchSpaceLatencyContainer.load_from_file(latency_file)

            latency_scale = max_latency(latency_container)

            latency_type = latency_container.latency_type
            constant_latency = latency_container.constant_latency / latency_scale
            operations_latencies = (np.array(latency_container.operations_latencies) / latency_scale).tolist()
            latency_container = SearchSpaceLatencyContainer(
                latency_type,
                constant_latency,
                operations_latencies,
            )

            search_space.apply_latency_container(latency_container)

            opt.max_latency_value /= latency_scale

    additional_kwargs = {}
    if is_search:

        # Use pretrain data loader for batch norm tuning (with all augmentations as during pre-train).
        bn_tune_loader = create_dataloader(
            data_dict['pretrain'], 640, batch_size // WORLD_SIZE * 2, gs, single_cls,
            hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect, rank=RANK,
            workers=workers, image_weights=opt.image_weights, quad=opt.quad,
            pad=0,
        )[0]

        # Define additional keyword arguments for ENOT optimizer which are necessary for yolov5 search procedure.
        additional_kwargs = {
            'bn_tune_dataloader': bn_tune_loader,
            'bn_tune_batches': 5,
            'bn_validation_tune_batches': 0,
            'sample_to_model_inputs': preprocess_data,
        }
        if opt.max_latency_value != 0.0:
            additional_kwargs['max_latency_value'] = opt.max_latency_value

    # Construct ENOT optimizer.
    optimizer = build_enot_optimizer(phase_name, model=model_to_optimize, optimizer=optimizer, **additional_kwargs)
    bn_tune_loader = BatchNormTuneDataLoaderWrapper(bn_tune_loader)  # Wrap to make PyTorch happy.

    # Perform output distribution optimization (recommended for convnets).
    if is_pretrain:
        sample_input = torch.zeros(1, 3, 256, 256).cuda()
        search_space.initialize_output_distribution_optimization(sample_input)

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = amp.GradScaler(enabled=cuda)
    compute_loss = ComputeLoss(model)  # init loss class
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights and not is_search:  # image_weights are not tested.
            # Generate indices
            if RANK in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if RANK != -1:
                indices = (torch.tensor(dataset.indices) if RANK == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if RANK != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw and not is_search:  # Warmup is not needed during search procedure
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale and not is_search:  # Multi-scale search is senseless. Run separate search runs instead.
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            loss, loss_items = None, None

            # Wrap forward+backward with closure.
            def closure():
                # Forward
                with amp.autocast(enabled=cuda):
                    model.model[-1].training = True
                    pred = model_to_optimize(imgs)  # forward

                    nonlocal loss, loss_items
                    loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                    if RANK != -1:
                        loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                    if opt.quad:
                        loss *= 4.

                    if is_search:  # For latency optimization.
                        loss = optimizer.modify_loss(loss)

                # Backward
                scaler.scale(loss).backward()

            # For gradient accumulation.
            optimizer.model_step(closure)

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model_to_optimize)
                last_opt_step = ni

            # Log
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.on_train_batch_end(ni, model, imgs, targets, paths, plots)
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        if is_search:  # Fix that during search we have only one parameter group.
            lr = [optimizer.param_groups[0]['lr'], 0.0, 0.0]
        scheduler.step()

        if RANK in [-1, 0]:
            # mAP
            callbacks.on_train_epoch_end(epoch=epoch)

            if ema is not None:
                ema.update_attr(model_to_optimize, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])

            if is_enot_phase:

                if is_pretrain:  # Testing baseline architecture during pretrain phase.
                    arch_to_test = [0] * len(search_space.search_blocks)
                else:  # Testing the best current architecture during search phase.
                    np.set_printoptions(precision=4, linewidth=120)
                    arch_to_test = optimizer.prepare_validation_model()
                    LOGGER.info(f'Testing architecture: {arch_to_test}')
                    LOGGER.info(f'It\'s latency: {current_latency(search_space)}')
                    LOGGER.info(
                        f'Architecture probabilities:\n'
                        f'{np.array(search_space.architecture_probabilities)}'
                    )

                    np.save(str(save_dir / f'p_{epoch + 1}.bin'), np.array(search_space.architecture_probabilities))
                    np.save(str(save_dir / f'a_{epoch + 1}.bin'), np.array(arch_to_test, dtype=np.int64))

                test_search_space = ema.ema if ema is not None else search_space  # Use EMA model if exists.
                test_model = test_search_space.get_network_by_indexes(arch_to_test).cuda()  # Extract single arch.
                tune_bn_stats(  # Separate batch norm tuning. This gives better network performance estimation.
                    test_model,
                    bn_tune_loader,
                    reset_bns=True,
                    set_momentums_none=True,
                    n_steps=50,  # You can reduce this value for faster execution.
                    sample_to_model_inputs=preprocess_data,
                )

            else:

                test_model = ema.ema if ema is not None else model

            final_epoch = epoch + 1 == epochs
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = val.run(data_dict,
                                           batch_size=batch_size // WORLD_SIZE * 2,
                                           imgsz=imgsz,
                                           model=test_model,
                                           single_cls=single_cls,
                                           dataloader=val_loader,
                                           save_dir=save_dir,
                                           save_json=is_coco and final_epoch,
                                           verbose=nc < 50 and final_epoch,
                                           plots=plots and final_epoch,
                                           callbacks=callbacks,
                                           compute_loss=compute_loss)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.on_fit_epoch_end(log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model_to_optimize)).half(),  # Save search space or model.
                        'ema': deepcopy(ema.ema).half() if ema is not None else None,
                        'updates': ema.updates if ema is not None else None,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None,
                        'phase_name': phase_name}  # Also save phase name to use in phase transitions.

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
                callbacks.on_model_save(last, epoch, final_epoch, best_fitness, fi)

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        if not evolve:
            # Not necessary to call it here.
            # if is_coco:  # COCO dataset
            #     for m in [last, best] if best.exists() else [last]:  # speed, mAP tests
            #         results, _, _ = val.run(data_dict,
            #                                 batch_size=batch_size // WORLD_SIZE * 2,
            #                                 imgsz=imgsz,
            #                                 model=attempt_load(m, device).half(),
            #                                 iou_thres=0.7,  # NMS IoU threshold for best pycocotools results
            #                                 single_cls=single_cls,
            #                                 dataloader=val_loader,
            #                                 save_dir=save_dir,
            #                                 save_json=True,
            #                                 plots=False)
            # Strip optimizers
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
        callbacks.on_train_end(last, best, plots, epoch)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--phase-name', type=str, default='train', help='ENOT phase name')
    parser.add_argument('--max-latency-value', type=float, default=0.0, help='Latency constraint for search procedure')
    parser.add_argument('--architecture-indices', type=str, default='[]', help='Train model architecture indices')
    parser.add_argument('--latency-file', type=str, default='', help='Path to search space latency container file')
    parser.add_argument('--grad-checkpoint', action='store_true', help='Activate gradient checkpointing')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    # Checks
    set_logging(RANK)
    if RANK in [-1, 0]:
        print(colorstr('train: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
        # check_git_status()  # We changed the repository!
        check_requirements(requirements=FILE.parent / 'requirements.txt', exclude=['thop'])

    # Resume
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = 'runs/evolve'
            opt.exist_ok = opt.resume
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        from datetime import timedelta
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        assert not opt.sync_bn, '--sync-bn known training issue, see https://github.com/ultralytics/yolov5/issues/3998'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device)
        if WORLD_SIZE > 1 and RANK == 0:
            _ = [print('Destroying process group... ', end=''), dist.destroy_process_group(), print('Done.')]

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        print(f'Hyperparameter evolution finished\n'
              f"Results saved to {colorstr('bold', save_dir)}\n"
              f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
