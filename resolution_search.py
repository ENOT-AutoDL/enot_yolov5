# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict
from typing import Tuple
from typing import Union

import enot
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from enot.experimental.resolution_search import ResolutionSearcherWithFixedLatencyIterator
from enot.experimental.resolution_search.resolution_strategy import ResolutionStrategy
from enot.latency import SearchSpaceLatencyContainer
from enot.latency import current_latency
from enot.models import SearchSpaceModel
from enot.models import SearchVariantsContainer
from enot.optimize import build_enot_optimizer
from enot.utils.batch_norm import BatchNormTuneDataLoaderWrapper
from enot.utils.batch_norm import tune_bn_stats
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch_optimizer import RAdam
from tqdm import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

import val  # for end-of-epoch mAP
from models.yolo import Model
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, init_seeds, \
    get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, set_logging, one_cycle, colorstr, methods
from utils.downloads import attempt_download
from utils.loss import ComputeLoss
from utils.torch_utils import select_device, intersect_dicts, torch_distributed_zero_first
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.metrics import fitness
from utils.loggers import Loggers
from utils.callbacks import Callbacks

LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(hyp,  # path/to/hyp.yaml or hyp dictionary
          opt,
          device,
          ):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    def preprocess_data(x: Tuple[torch.Tensor]) -> Tuple[Tuple[torch.Tensor], Dict]:
        return (x[0].to(device).float() / 255.0, ), {}

    phase_name = opt.phase_name
    if phase_name != 'search':
        raise RuntimeError('This file works only with search procedure.')

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

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(RANK):
            weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint

        # We select EMA model for search procedure to compare models closer to production deployment.
        load_ema_from_checkpoint = ckpt['phase_name'] == 'pretrain' and phase_name == 'search' and 'ema' in ckpt
        ckpt_model = ckpt['model']
        if load_ema_from_checkpoint:
            LOGGER.info('Using EMA checkpoint for search')
            ckpt_model = ckpt['ema']
        ckpt_config = ckpt_model.original_model.yaml if isinstance(ckpt_model, SearchSpaceModel) else ckpt_model.yaml
        model = Model(cfg or ckpt_config, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt_model.float().state_dict()  # checkpoint state_dict as FP32
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    model_to_optimize: Union[nn.Module, SearchSpaceModel] = model
    if any(isinstance(layer, SearchVariantsContainer) for layer in model.modules()):
        model_to_optimize = search_space = SearchSpaceModel(model).to(device)
    else:
        raise ValueError(f'Phase "{phase_name}" requires model yaml file to contain search blocks')

    # load weights only after SearchSpaceModel is applied to model class
    if pretrained:
        csd = intersect_dicts(csd, model_to_optimize.state_dict(), exclude=exclude)  # intersect
        model_to_optimize.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'Transferred {len(csd)}/{len(model_to_optimize.state_dict())} items from {weights}')  # report

    # Freeze
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False

    # Optimizer
    params = list(search_space.architecture_parameters())
    optimizer = RAdam(params, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with {len(params)} search blocks")
    del params

    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']

    # Resume
    if pretrained and phase_name == ckpt['phase_name']:
        raise RuntimeError('Resume is not supported')

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

    bn_tune_loader = create_dataloader(
        data_dict['pretrain'], imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
        hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect, rank=RANK,
        workers=workers, image_weights=opt.image_weights, quad=opt.quad,
        pad=0,
    )[0]
    bn_tune_loader = BatchNormTuneDataLoaderWrapper(bn_tune_loader)

    additional_kwargs = {
        'bn_tune_dataloader': bn_tune_loader,
        'bn_tune_batches': 2,
        'bn_validation_tune_batches': 0,
        'sample_to_model_inputs': preprocess_data,
    }
    if opt.max_latency_value != 0.0:
        additional_kwargs['max_latency_value'] = opt.max_latency_value

    enot_optimizer = build_enot_optimizer(phase_name, model=model_to_optimize, optimizer=optimizer, **additional_kwargs)

    def resolution_strategy_constructor(r: int) -> ResolutionStrategy:

        class YoloDummyResolutionStrategy(ResolutionStrategy):
            def __call__(self, dataloader, *args, **kwargs):
                yield from dataloader

        return YoloDummyResolutionStrategy(None)

    search_resolution_iter = ResolutionSearcherWithFixedLatencyIterator(
        enot_optimizer=enot_optimizer,
        dataloader=bn_tune_loader,
        latency_type='mmac.fvcore',  # This will not be used.
        min_resolution=224,
        max_resolution=640,
        resolution_tolerance=gs - 1,  # To guarantee that we will eventually visit all resolutions near optimal.
        resolution_strategy_constructor=resolution_strategy_constructor,
        sample_to_model_inputs=preprocess_data,
        do_fix_resolution=False,
    )

    base_save_dir = save_dir

    for r_step, (resolution, _) in enumerate(search_resolution_iter):

        # Equivalent to round to the nearest gs multiple.
        resolution = int(round(resolution / gs)) * gs
        imgsz = resolution

        # Set new save directory for each resolution search iteration.
        save_dir = Path(base_save_dir) / f'resolution_step_{r_step + 1}'

        # Create new callbacks each iteration.
        callbacks = Callbacks()

        logger = enot.logging.prepare_log(save_dir / 'log.txt', logger_name='resolution_search_logger')
        logger.info(f'Running resolution search step #{r_step} at resolution {imgsz}.')

        # Create new loggers each iteration.
        if RANK in [-1, 0]:
            loggers = Loggers(save_dir, weights, opt, hyp, logger)  # loggers instance

            # Register actions
            for k in methods(loggers):
                callbacks.register_action(k, callback=getattr(loggers, k))

        start_epoch, best_fitness, best_map, best_architecture = 0, 0.0, 0.0, None

        # Create new scheduler each iteration.
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

        # Create new train loader each iteration.
        train_loader, dataset = create_dataloader(train_path, resolution, batch_size // WORLD_SIZE, gs, single_cls,
                                                  hyp=hyp, augment=False, cache=opt.cache, rect=True, rank=RANK,
                                                  workers=workers, image_weights=opt.image_weights, quad=opt.quad,
                                                  pad=0.5, prefix=colorstr(f'{phase_name}: '))

        mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
        nb = len(train_loader)  # number of batches
        assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

        # Process 0
        if RANK in [-1, 0]:
            val_loader = create_dataloader(val_path, resolution, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                           hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
                                           workers=workers, pad=0.5,
                                           prefix=colorstr('val: '))[0]

            callbacks.on_pretrain_routine_end()

        latency_folder = opt.latency_file
        if latency_folder == '':
            raise ValueError('Resolution search requires specifying latency file')

        latency_container = SearchSpaceLatencyContainer.load_from_file(Path(latency_folder) / f'r{resolution}.pkl')
        search_space.apply_latency_container(latency_container)

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
        results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
        scheduler.last_epoch = start_epoch - 1  # do not move
        scaler = amp.GradScaler(enabled=cuda)
        compute_loss = ComputeLoss(model)  # init loss class
        logger.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                    f'Using {train_loader.num_workers} dataloader workers\n'
                    f'Logging results to {save_dir}\n'
                    f'Starting training for {epochs} epochs...')
        for epoch in range(start_epoch, epochs):  # epoch --------------------------------------------------------------
            model.train()

            mloss = torch.zeros(3, device=device)  # mean losses
            if RANK != -1:
                train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(train_loader)
            logger.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
            if RANK in [-1, 0]:
                pbar = tqdm(pbar, total=nb)  # progress bar
            enot_optimizer.zero_grad()
            for i, (imgs, targets, paths, _) in pbar:  # batch ---------------------------------------------------------
                ni = i + nb * epoch  # number integrated batches (since train start)
                imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

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

                        # For latency optimization.
                        loss = enot_optimizer.modify_loss(loss)

                    # Backward
                    scaler.scale(loss).backward()

                # For gradient accumulation.
                enot_optimizer.model_step(closure)

                # Optimize
                scaler.step(enot_optimizer)  # optimizer.step
                scaler.update()
                enot_optimizer.zero_grad()

                # Log
                if RANK in [-1, 0]:
                    mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                    mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                    pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                        f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                    callbacks.on_train_batch_end(ni, model, imgs, targets, paths, plots)
                # end batch --------------------------------------------------------------------------------------------

            # Scheduler
            # Fix that during search we have only one parameter group.
            lr = [enot_optimizer.param_groups[0]['lr'], 0.0, 0.0]  # for loggers
            scheduler.step()

            if RANK in [-1, 0]:
                # mAP
                callbacks.on_train_epoch_end(epoch=epoch)

                # Testing the best current architecture during search phase.
                np.set_printoptions(precision=4, linewidth=120)
                arch_to_test = enot_optimizer.prepare_validation_model()
                logger.info(f'Testing architecture: {arch_to_test}')
                logger.info(f'It\'s latency: {current_latency(search_space)}')
                logger.info(f'Architecture probabilities:\n{np.array(search_space.architecture_probabilities)}')

                test_model = search_space.get_network_by_indexes(arch_to_test).cuda()  # Extract single arch.
                tune_bn_stats(  # Separate batch norm tuning. This gives better network performance estimation.
                    test_model,
                    bn_tune_loader,
                    reset_bns=True,
                    set_momentums_none=True,
                    n_steps=250,  # You can reduce this value for faster execution.
                    sample_to_model_inputs=preprocess_data,
                )

                final_epoch = epoch + 1 == epochs
                if not noval or final_epoch:  # Calculate mAP
                    results, maps, _ = val.run(data_dict,
                                               batch_size=batch_size // WORLD_SIZE * 2,
                                               imgsz=imgsz,
                                               model=test_model,
                                               single_cls=single_cls,
                                               dataloader=val_loader,
                                               save_dir=save_dir,
                                               save_json=False,
                                               verbose=False,
                                               plots=False,
                                               callbacks=callbacks,
                                               compute_loss=compute_loss)

                # Update best mAP
                fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                if fi > best_fitness:
                    best_fitness = fi
                if results[3] > best_map:
                    best_map = results[3]
                    best_architecture = arch_to_test
                log_vals = list(mloss) + list(results) + lr
                callbacks.on_fit_epoch_end(log_vals, epoch, best_fitness, fi)

            # end epoch ------------------------------------------------------------------------------------------------
        # end training -------------------------------------------------------------------------------------------------
        if RANK in [-1, 0]:
            logger.info(
                f"Overall iteration best mAP is {best_map * 100:.2f} "
                f"for architecture {best_architecture} (r={imgsz})"
            )
            logger.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
            callbacks.on_train_end(last, best, plots, epoch)
            logger.info(f"Results saved to {colorstr('bold', save_dir)}")

        search_resolution_iter.set_resolution_target_metric(best_map)

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
        check_git_status()
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
    train(opt.hyp, opt, device)
    if WORLD_SIZE > 1 and RANK == 0:
        _ = [print('Destroying process group... ', end=''), dist.destroy_process_group(), print('Done.')]


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
