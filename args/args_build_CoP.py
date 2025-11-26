import os
import yaml
import argparse
from datetime import datetime, timedelta, timezone
import shutil
import random
import torch

from utils.common_utils import copy_source,getLogger


def get_args_parser(appointed_cfg=None):
    
    parser = argparse.ArgumentParser()
     
    parser.add_argument('--cfg', type=str, 
                        default="config/config.yml",
                        help="config file path")
    parser.add_argument('--notes', type=str, 
                        default="N")
    
    args = parser.parse_args()
    

    #################################
    # Read yaml file to update args #
    #################################
    if appointed_cfg is not None:
        args.cfg = appointed_cfg
    with open(args.cfg, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    for k, v in cfg.items():
        parser.add_argument('--{}'.format(k), default=v, type=type(v))
    args = parser.parse_args()
    
    ###############################################
    # Dynamically modify some args here if needed #
    ###############################################
    if args.num_workers == -1:
        args.num_workers = torch.get_num_threads() - 1

    ################################
    # Modify random seed if needed #
    ################################
    if args.seed < 0:
        args.seed = random.randint(0, 1000000)
        
    #################################
    # Setup log and checkpoint path #
    #################################
    if args.curr_time == -1:
        beijing_tz = timezone(timedelta(hours=8))
        current_time = datetime.now(beijing_tz)
        args.curr_time = current_time.strftime("%Y-%m-%d_%H:%M:%S")
    
    
    os.makedirs(args.log_dir, exist_ok=True)
   
 
            
    
    return args

