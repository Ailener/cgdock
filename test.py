import numpy as np
import os
import torch
from data import get_data
from torch_geometric.loader import DataLoader
from utils.metrics import *
from utils.utils import *
from datetime import datetime
from utils.logging_utils import Logger
import sys
import argparse
import random
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
import shlex
from utils.parse import parse_args

args_new = parse_args()
command = shlex.split(command)

args.local_eval = args_new.local_eval
args.ckpt = args_new.ckpt   
args.data_path = args_new.data_path
args.resultFolder = args_new.resultFolder  
args.seed = args_new.seed
args.exp_name = args_new.exp_name    
args.batch_size = args_new.batch_size
args.tqdm_interval = 0.1
args.disable_tqdm = False

set_seed(args.seed)

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision)

pre = f"{args.resultFolder}/{args.exp_name}"

accelerator.wait_for_everyone()

os.makedirs(pre, exist_ok=True)
logger = Logger(accelerator=accelerator, log_path=f'{pre}/test.log')
logger.log_message(f"{' '.join(sys.argv)}")
torch.multiprocessing.set_sharing_strategy('file_system')

if args.redocking:
    args.compound_coords_init_mode = "redocking"
elif args.redocking_no_rotate:
    args.redocking = True
    args.compound_coords_init_mode = "redocking_no_rotate"

train, valid, test= get_data(args, logger, addNoise=args.addNoise, use_whole_protein=args.use_whole_protein, compound_coords_init_mode=args.compound_coords_init_mode, pre=args.data_path)
logger.log_message(f"data point train: {len(train)}, valid: {len(valid)}, test: {len(test)}")
num_workers = 10

test_loader = DataLoader(test, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=num_workers)
test_unseen_pdb_list = [line.strip() for line in open('./split_pdb_id/unseen_test_index')]

test_unseen_index = test.data.query("(group =='test') and (pdb in @test_unseen_pdb_list)").index.values
test_unseen_index_for_select = np.array([np.where(test._indices == i) for i in test_unseen_index]).reshape(-1)
test_unseen = test.index_select(test_unseen_index_for_select)
test_unseen_loader = DataLoader(test_unseen, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, pin_memory=False, num_workers=10)

from models.model import *
device = 'cuda'

model = get_model(args, logger)
model = accelerator.prepare(model)

from safetensors.torch import load_file
model.load_state_dict(load_file(args.ckpt))

if args.pred_dis:
    criterion = nn.MSELoss()
    pred_dis = True
else:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.posweight))

if args.coord_loss_function == 'MSE':
    com_coord_criterion = nn.MSELoss()
elif args.coord_loss_function == 'SmoothL1':
    com_coord_criterion = nn.SmoothL1Loss()

if args.pocket_cls_loss_func == 'bce':
    pocket_cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')

pocket_cls_criterion = FocalBCEWithLogitsLoss(gamma=2.0)  
pocket_coord_criterion = nn.HuberLoss(delta=args.pocket_coord_huber_delta)
pocket_radius_criterion = nn.HuberLoss(delta=args.pocket_coord_huber_delta)

model.eval()

logger.log_message(f"Begin test")
if accelerator.is_main_process:
    metrics = evaluate_mean_pocket_cls_coord_multi_task(accelerator, args, test_loader, accelerator.unwrap_model(model), com_coord_criterion, criterion, pocket_cls_criterion, pocket_coord_criterion, pocket_radius_criterion, args.relative_k,
                                                        accelerator.device, pred_dis=pred_dis, use_y_mask=False, stage=2)
    logger.log_stats(metrics, 0, args, prefix="Test_all")

    metrics = evaluate_mean_pocket_cls_coord_multi_task(accelerator, args, test_unseen_loader, accelerator.unwrap_model(model), com_coord_criterion, criterion, pocket_cls_criterion, pocket_coord_criterion, pocket_radius_criterion, args.relative_k,
                                                        accelerator.device, pred_dis=pred_dis, use_y_mask=False, stage=2)
    logger.log_stats(metrics, 0, args, prefix="Test_unseen")
accelerator.wait_for_everyone()
