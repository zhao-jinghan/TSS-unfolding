import os
import shutil
import numpy as np
import pickle
import yaml
import multiprocessing
from utils.common_utils import *
from datasets.build_knowledge.helper import *


def _save_path(args,logger=None): 
    pseudo_dir = os.path.join(args.pwd,'pseudo')
    os.makedirs(pseudo_dir,exist_ok=True)

    config = args.stateVNM_config
    param = None
    if config['criteria'] == 'threshold':
        param = str(config['threshold'])
    elif config['criteria'] == 'topK':
        param = str(config['topK'])
    elif config['criteria'] == 'threshold+topK':
        param = str(config['threshold'])+'_'+str(config['topK'])
    filename = f"stateVNM_{config['criteria']}_{param}.pkl" 
    save_path = os.path.join(pseudo_dir,filename) 
    return save_path

def _pseudo_per_shot(args,logger,state_simscore,state_type,state2node):
    node_scores = np.zeros((args.num_nodes))
    config = args.stateVNM_config

    mask = state_simscore > config['threshold'] if 'threshold' in config['criteria'] else state_simscore > 0

    for i,mask_i in enumerate(mask):
        if not mask_i:
            continue
        
        state_id = i + state_type * args.num_steps 
        node_id = state2node[state_id]
        
        node_scores[node_id] = max(node_scores[node_id], state_simscore[i])
    
    idx,scores = find_matching_of_a_segment(
        node_scores, 
        criteria=config['criteria'], 
        threshold=config['threshold'],
        topK=config['topK'])

    return idx,scores

def load_stateVNM_labels(args,logger):
    save_path = _save_path(args,logger)
    return pickle.load(open(save_path,'rb'))

def judge_state_type(sim):
    pass

def _pseudo_per_video(args,logger,video_id,
                      sim_dir,state2node_byType,
                      stat_logger=None):
    sim_score = np.load(os.path.join(sim_dir,f'{video_id}.npy'))
    num_shots = sim_score.shape[0]
    
    _pseudo_video = []
    for i in range(num_shots):
        state_type = judge_state_type(sim_score[i])
    
        idx,scores = _pseudo_per_shot(args,logger,
                                    state_simscore=sim_score[i][state_type], # (num_steps,)
                                    state_type=state_type,
                                    state2node=state2node_byType[state_type])

        _pseudo_video.append({
            'type':state_type,
            'idx':idx,
            'scores':scores
        })

        if stat_logger is not None:
            stat_logger.update('stateVNM_cnt',len(idx))
            stat_logger.update('stateVNM_score',scores)
    return video_id,_pseudo_video
    
def unpack_param(param_tuple): 
    return _pseudo_per_video(*param_tuple)

from datasets.build_CoP.state_level.get_state_nodes import load_nodes
    
def pseudo_label_stateVNM(args,logger,sim_dir):
    save_path = _save_path(args,logger)
    assert not os.path.exists(save_path), "pseudo label already exists"

    state2node_byType = [load_nodes(args,logger,state_type=i)[1] for i in range(3)]
    
    
    buffer = {} 
    buffer_limit = 5000 
    for filename in tqdm(os.listdir(sim_dir)):
        if not filename.endswith('.npy'):
            continue
        video_id = filename.split('.')[0]
        video_id,pseudo_data = _pseudo_per_video(args,logger,video_id,
                        sim_dir,state2node_byType,
                        )
        buffer[video_id] = pseudo_data
        
        if len(buffer) >= buffer_limit:
            flush_buffer_to_disk(buffer,save_path,logger)
        
    flush_buffer_to_disk(buffer,save_path,logger)