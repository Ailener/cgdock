import numpy as np
import os
import sys
import argparse
import random
import torch
from tqdm.auto import tqdm
from data import get_data
from torch_geometric.loader import DataLoader
from utils.metrics import *
from utils.utils import *
from datetime import datetime
from utils.logging_utils import Logger
from utils.parse import parse_args
from torch.utils.data import RandomSampler
from torch_scatter import scatter_mean
from utils.metrics_to_tsb import metrics_runtime_no_prefix
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def init_accelerator(args):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=args.mixed_precision
    )
    set_seed(args.seed)
    torch.multiprocessing.set_sharing_strategy('file_system')
    return accelerator

def init_logging(args, accelerator):
    pre = f"{args.resultFolder}/{args.exp_name}"

    if accelerator.is_main_process:
        os.makedirs(f"{pre}/models", exist_ok=True)

        writers = {}
        if not args.disable_tensorboard:
            tsb_dir = f"{pre}/tsb_runtime"
            os.makedirs(tsb_dir, exist_ok=True)
            writers = {
                "train": SummaryWriter(f"{tsb_dir}/train"),
                "valid": SummaryWriter(f"{tsb_dir}/valid"),
                "test": SummaryWriter(f"{tsb_dir}/test"),
                "test_pp": SummaryWriter(f"{tsb_dir}/test_use_predicted_pocket"),
            }
    else:
        writers = None

    accelerator.wait_for_everyone()

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    logger = Logger(accelerator, log_path=f"{pre}/{timestamp}.log")
    logger.log_message(" ".join(sys.argv))

    return logger, writers, pre

def build_dataloaders(args, logger):
    if args.redocking:
        args.compound_coords_init_mode = "redocking"
    elif args.redocking_no_rotate:
        args.redocking = True
        args.compound_coords_init_mode = "redocking_no_rotate"

    train, valid, test = get_data(
        args, logger,
        addNoise=args.addNoise,
        use_whole_protein=args.use_whole_protein,
        compound_coords_init_mode=args.compound_coords_init_mode,
        pre=args.data_path
    )

    logger.log_message(
        f"data point train: {len(train)}, valid: {len(valid)}, test: {len(test)}"
    )

    num_workers = 0
    if args.sample_n > 0:
        sampler = RandomSampler(train, replacement=True, num_samples=args.sample_n)
        train_loader = DataLoader(
            train, batch_size=args.batch_size,
            sampler=sampler, follow_batch=['x', 'compound_pair']
        )
    else:
        train_loader = DataLoader(
            train, batch_size=args.batch_size,
            shuffle=True, follow_batch=['x', 'compound_pair']
        )

    valid_loader = DataLoader(valid, batch_size=args.batch_size, shuffle=False,
                              follow_batch=['x', 'compound_pair'])
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                             follow_batch=['x', 'compound_pair'])

    return train_loader, valid_loader, test_loader

def build_model_and_optim(args, logger, train_loader):
    model = get_model(args, logger)

    if args.optim == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch

    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.5, end_factor=1.0, total_iters=warmup_steps
    )

    scheduler_post = build_post_scheduler(args, optimizer, steps_per_epoch)

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_post],
        milestones=[warmup_steps]
    )

    return model, optimizer, scheduler

def build_post_scheduler(args, optimizer, steps_per_epoch):
    total_steps = (args.total_epochs - args.warmup_epochs) * steps_per_epoch

    if args.lr_scheduler == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, 1.0, total_steps)
    if args.lr_scheduler == "poly_decay":
        return torch.optim.lr_scheduler.LinearLR(optimizer, 1.0, 0.0, total_steps)
    if args.lr_scheduler == "exp_decay":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    if args.lr_scheduler == "cosine_decay":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=1e-5
        )
    if args.lr_scheduler == "cosine_decay_restart":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, eta_min=1e-4
        )


def build_criterions(args):
    criterions = {}

    criterions["pair"] = (
        nn.MSELoss() if args.pred_dis
        else nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.posweight))
    )

    criterions["coord"] = (
        nn.MSELoss() if args.coord_loss_function == "MSE"
        else nn.SmoothL1Loss()
    )

    criterions["pocket_cls"] = FocalBCEWithLogitsLoss(gamma=2.0) 
    criterions["pocket_coord"] = nn.HuberLoss(delta=args.pocket_coord_huber_delta)
    criterions["pocket_radius"] = nn.HuberLoss(delta=args.pocket_coord_huber_delta)

    return criterions

def train_one_epoch(
    epoch, model, loader, optimizer, scheduler,
    criterions, accelerator, args, logger, writers
):
    model.train()
    stats = init_epoch_stats()

    data_iter = loader if args.disable_tqdm else tqdm(
        loader, mininterval=args.tqdm_interval,
        disable=not accelerator.is_main_process
    )

    for step, data in enumerate(data_iter, 1):
        loss, batch_stats = train_step(
            model, data, criterions, optimizer, scheduler,
            accelerator, args
        )
        accumulate_stats(stats, batch_stats)

        if step % args.log_interval == 0:
            logger.log_stats(batch_stats, epoch, args, prefix="train")

    epoch_metrics = finalize_epoch_stats(stats)
    logger.log_stats(epoch_metrics, epoch, args, prefix="Train")

    if accelerator.is_main_process and writers:
        metrics_runtime_no_prefix(epoch_metrics, writers["train"], epoch)

def train_step(model, data, criterions, optimizer, scheduler, accelerator, args):
    optimizer.zero_grad()

    outputs = model(data, train=True)
    (
        com_coord_pred, compound_batch,
        y_pred, y_pred_by_coord,
        pocket_cls_pred, pocket_cls,
        protein_mask, _, pocket_center_pred,
        dis_map, keepNode_less_5, pocket_radius_pred
    ) = outputs

    com_coord = data.coords

    loss_dict = compute_losses(
        args, criterions,
        com_coord_pred, com_coord,
        y_pred, y_pred_by_coord, dis_map,
        pocket_cls_pred, pocket_cls,
        pocket_center_pred, data.coords_center,
        pocket_radius_pred, data.ligand_radius,
        protein_mask
    )

    loss = sum(loss_dict.values())
    accelerator.backward(loss)

    if args.clip_grad and accelerator.sync_gradients:
        accelerator.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()
    scheduler.step()

    return loss, loss_dict
    
def run_validation_and_test(
    epoch, model, valid_loader, test_loader,
    criterions, accelerator, args, logger, writers, pre
):
    model.eval()

    if not args.disable_validate:
        metrics = evaluate_mean_pocket_cls_coord_multi_task(
            accelerator, args, valid_loader, model,
            criterions["coord"], criterions["pair"],
            criterions["pocket_cls"], criterions["pocket_coord"],
            criterions["pocket_radius"], args.relative_k,
            accelerator.device, pred_dis=args.pred_dis, stage=1
        )
        logger.log_stats(metrics, epoch, args, prefix="Valid")
        if writers:
            metrics_runtime_no_prefix(metrics, writers["valid"], epoch)

    for stage, tag in [(1, "test"), (2, "test_pp")]:
        metrics = evaluate_mean_pocket_cls_coord_multi_task(
            accelerator, args, test_loader,
            accelerator.unwrap_model(model),
            criterions["coord"], criterions["pair"],
            criterions["pocket_cls"], criterions["pocket_coord"],
            criterions["pocket_radius"], args.relative_k,
            accelerator.device, pred_dis=args.pred_dis, stage=stage
        )
        logger.log_stats(metrics, epoch, args, prefix=tag.upper())
        if writers:
            metrics_runtime_no_prefix(metrics, writers[tag], epoch)

    accelerator.save_state(f"{pre}/models/epoch_{epoch}")
    accelerator.save_state(f"{pre}/models/epoch_last")


def main():
    args = parse_args()
    accelerator = init_accelerator(args)
    logger, writers, pre = init_logging(args, accelerator)

    train_loader, valid_loader, test_loader = build_dataloaders(args, logger)

    model, optimizer, scheduler = build_model_and_optim(args, logger, train_loader)
    criterions = build_criterions(args)

    last_epoch = maybe_resume(accelerator, scheduler, args, pre, logger)

    for epoch in range(last_epoch + 1, args.total_epochs):
        train_one_epoch(
            epoch, model, train_loader, optimizer, scheduler,
            criterions, accelerator, args, logger, writers
        )

        if accelerator.is_main_process:
            run_validation_and_test(
                epoch, model, valid_loader, test_loader,
                criterions, accelerator, args, logger, writers, pre
            )







    
   

