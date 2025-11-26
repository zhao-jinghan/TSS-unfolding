import os
import yaml
import shutil
import argparse
from datetime import datetime, timedelta, timezone
import random
import torch

from utils.common_utils import copy_source,getLogger


def get_args_parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--cfg', type=str, 
                        default="config/config.yml",
                        help="config file path")
    parser.add_argument('--notes', type=str, 
                        default="N")
    
    parser.add_argument("--checkpoint", type=str, 
                        required=False,
                        help="a path to model checkpoint file to load pretrained weights")
    
    parser.add_argument('--use_wandb', type=int, 
                        default=0,
                        help="1 means use wandb to log experiments, 0 otherwise")
    
    parser.add_argument('--downstream_dataset_name', type=str,
                        default="COIN")
    
    parser.add_argument('--downstream_task_name', type=str,
                        default="task_cls")
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
        
    if not torch.cuda.is_available():
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(args.device)
        
    args.working_abspath = os.path.abspath('./')

    if args.seed < 0:
        args.seed = random.randint(0, 1000000)
   
    return args 

