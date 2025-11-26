import os
import pdb
import sys
from pathlib import Path 
import pickle
import numpy as np
import random
import copy
import json
import time
import glob
import yaml
from utils.common_utils import Bunch
from tqdm import tqdm
from collections import defaultdict
from scipy.sparse import csr_matrix

import torch
from torch.utils.data import Dataset
import re 
from utils.common_utils import numpy_topk_indices, minmax_normalize, need_logging


def parse_obj(str):
    match = re.match(r'^[a-z]+', str)
    if match:
        return match.group(0) 
    else:
        return "" 
from datasets.build_knowledge.get_samples import load_samples_new
class HT100M(Dataset):
    def __init__(self, args, logger):
        
        self.args, self.logger = args, logger
        
        if (not hasattr(args, 'partition_dataset')) or (
            not args.partition_dataset) or (
            not args.use_ddp):
            
            if args.target_stage not in ['task','step','state']:
                self.samples,self.feats_all = load_samples_new(args,logger,'step')
            else:
                self.samples,self.feats_all = load_samples_new(args,logger,args.target_stage)
            
        else:
            with open(os.path.join( 
                args.dataset_dir['ht100m'], 'feats', 
                'feats_all-mean_agg-rank_{}-of-{}.pickle'.format(
                    args.rank, args.world_size)), 'rb') as f:
                self.feats_all = pickle.load(f)

        config = args.graph_task_config 
        with open(config,'rb') as f:
            config = yaml.safe_load(f)
            config = Bunch(config)
        
        if 'taskVNM' in args.adapter_objective:
            from datasets.build_CoP.task_level.pseudo_label_taskVNM import load_taskVNM_labels
            self.pseudo_labels_taskVNM = load_taskVNM_labels(config,logger)
        config = args.graph_step_config 
        with open(config,'rb') as f:
            config = yaml.safe_load(f)
            config = Bunch(config)
       
        if 'stepVNM' in args.adapter_objective:
            from datasets.build_CoP.step_level.pseudo_label_stepVNM import load_stepVNM_labels
            self.pseudo_labels_stepVNM = load_stepVNM_labels(config,logger)
        if 'stepTCL' in args.adapter_objective:
            from datasets.build_CoP.step_level.pseudo_label_stepTCL import load_stepTCL_labels
            self.pseudo_labels_stepTCL = load_stepTCL_labels(config,logger)
        if 'stepNRL' in args.adapter_objective:
            from datasets.build_CoP.step_level.pseudo_label_stepNRL import load_stepNRL_labels
            self.pseudo_labels_stepNRL = load_stepNRL_labels(config,logger)
        config = args.graph_state_config 
        with open(config,'rb') as f:
            config = yaml.safe_load(f)
            config = Bunch(config)
        if 'stateVNM' in args.adapter_objective:
            from datasets.build_CoP.state_level.pseudo_label_stateVNM import load_stateVNM_labels
            self.pseudo_labels_stateVNM = load_stateVNM_labels(config,logger)

    def __len__(self):
        return len(self.videos)

                   
    def parse_all_pseudo_labels(self,index):
        video_id,shot_idx = self.samples[index]

        args = self.args 
       
        total_pseudo_labels = []
        for obj in args.adapter_objective:
            pseudo_labels = getattr(self, f'pseudo_labels_{obj}')
            indices = pseudo_labels[video_id][shot_idx]['idx']

            stage = parse_obj(obj)
            num_nodes = getattr(self.args,f'{stage}_num_nodes')
            pseudo_label = torch.zeros(num_nodes)

            for i in range(min(len(indices), args.cls_topK[obj] )): 
                pseudo_label[indices[i]] = 1
            total_pseudo_labels.append(pseudo_label)
    
        return total_pseudo_labels
   

    def __getitem__(self, index):
        video_id,shot_idx = self.samples[index]
        segment_video_feat = np.load(self.feats_all[video_id])[shot_idx]

        pseudo_label_list = self.parse_all_pseudo_labels(index)
        return tuple([torch.FloatTensor(segment_video_feat)]+ pseudo_label_list) 

