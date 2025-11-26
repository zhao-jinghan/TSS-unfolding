import os
import yaml
import argparse
from datetime import datetime, timedelta, timezone
import random
import shutil

import torch

from utils.common_utils import copy_source, init_distributed, is_main_process


def get_args_parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cfg', type=str, 
                        default="configs/CoP_pretrain/CoP_pretrain_by_stage.yml",
                        help="config file path")
    
    parser.add_argument("--load_pretrained", type=int, 
                        required=False,
                        help="whether to load pretrained weights for training")
    
    parser.add_argument("--checkpoint", type=str, 
                        required=False,
                        help="a path to model checkpoint file to load pretrained weights")
    parser.add_argument("--target_stage", type=str, 
                        required=False,
                        help="pretrain by stage ")
    
    parser.add_argument('--use_wandb', type=int, 
                        default=0,
                        help="1 means use wandb to log experiments, 0 otherwise")
    
    parser.add_argument('--use_ddp', type=int, 
                        default=0,
                        help="1 means use pytorch GPU parallel distributed training, 0 otherwise")
    parser.add_argument('--use_amp', type=bool, 
                        default=False)
    
    parser.add_argument('--notes', type=str, 
                        default="N")
    
    parser.add_argument('--local-rank','--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    
    
    #################################
    # Read yaml file to update args #
    #################################
    with open(args.cfg, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    for k, v in cfg.items():
        parser.add_argument('--{}'.format(k), default=v, type=type(v))
    args = parser.parse_args()
    
    
    #################################
    # Setup log and checkpoint path #
    #################################
    
    if args.curr_time == -1:
        beijing_tz = timezone(timedelta(hours=8))
        current_time = datetime.now(beijing_tz)
        args.curr_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")
    
    args.checkpoint_dir = os.path.abspath(
            os.path.join(args.checkpoint_dir, f'{args.curr_time}_{args.exp_name}_{args.notes}'))
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    os.makedirs(args.log_dir, exist_ok=True)

    
            
        
    ###############################################
    # Dynamically modify some args here if needed #
    ###############################################
    if args.num_workers == -1:
        args.num_workers = torch.get_num_threads() - 1
        print(f'num_workers: {args.num_workers}','-'*20)
    
    
    args.working_abspath = os.path.abspath('./')
    
    if args.seed < 0:
        args.seed = random.randint(0, 1000000)
        
        
    ###############
    # Setup DDP #
    ###############
    if args.use_ddp:
        init_distributed()
        args.device = torch.device('cuda')
        args.world_size = torch.cuda.device_count()
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        if not torch.cuda.is_available():
            args.device = torch.device('cpu')
        else:
            args.device = torch.device(args.device)

            
    ###############
    # Setup wandb #
    ###############
    
    
    if args.pretrain_knowledge == 'CoP' and args.target_stage is not None:
        print(args.adapter_objective,args.target_stage)
        if args.target_stage not in args.adapter_objective:
            _all_obj = []
            for stage in args.target_stage.split('_'):
                _all_obj += args.adapter_objective[stage]
            
            args.adapter_objective = _all_obj
        else:
            args.adapter_objective = args.adapter_objective[args.target_stage]

    return args

