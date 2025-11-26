import os
import numpy as np
from utils.common_utils import *
from datasets.build_knowledge.helper import *
import pickle
from multiprocessing import Pool
from tqdm import tqdm

def _save_path(args,logger=None):
    pseudo_dir = os.path.join(args.pwd,'pseudo')
    config = args.stepTCL_config
    param = None
    if config['criteria'] == 'threshold':
        param = str(config['threshold'])
    elif config['criteria'] == 'topK':
        param = str(config['topK'])
    elif config['criteria'] == 'threshold+topK':
        param = str(config['threshold'])+'_'+str(config['topK'])

    filename = f"stepTCL_{config['criteria']}_{param}.pkl"
    save_path = os.path.join(pseudo_dir,filename)
    return save_path

def _pseudo_per_shot(args,logger,VTM,step_task_occurrence,step2node):
    candidates = np.zeros(args.num_nodes) 
    for task_id in VTM: 
        step_list = np.where(step_task_occurrence[:,task_id] > 0)[0]
        for step_id in step_list: 
            node_id = step2node[step_id]
            candidates[node_id] = np.max([candidates[node_id],step_task_occurrence[step_id,task_id]])
    config = args.stepTCL_config 
    idx, scores = find_matching_of_a_segment(
        candidates,
        criteria=config['criteria'],
        threshold=config['threshold'],
        topK=config['topK']
    )
    return idx,scores
def unpack_param(param_tuple): 
    return _pseudo_per_video(*param_tuple)

def load_stepTCL_labels(args,logger):
    save_path = _save_path(args,logger)
    return pickle.load(open(save_path,'rb'))

def _pseudo_per_video(args,logger, video_id, VTM, step_task_occurrence,step2node,stat_logger=None):
    num_shots = len(VTM)
    _pseudo_video = []
    
    for i in range(num_shots):
        idx, scores = _pseudo_per_shot(
            args, logger,
            VTM[i]['idx'],
            step_task_occurrence,step2node
        )
        _pseudo_video.append({'idx': idx, 'scores': scores})
    
    if stat_logger is not None:
        stat_logger.update('stepTCL_cnt',len(idx))
        stat_logger.update('stepTCL_score',scores)
    return video_id,_pseudo_video

def pseudo_label_stepTCL(args, logger):
    save_path = _save_path(args,logger)
    assert not os.path.exists(save_path), "pseudo label already exists"
    stat_logger = Statistic_Logger() # for statistics
    
    # 1. prepare params 
    path_step2node = os.path.join(args.pwd,'node_label','step2node.pkl')
    step2node  = pickle.load(open(path_step2node, 'rb'))

    VTM = load_stepVTM_labels(args,logger) # TODO 

    step_task_occurrence = pickle.load(open(os.path.join(args.pwd,'step_task_occurrence.pkl'),'rb'))
    
    buffer = {}
    buffer_limit = 5000
    for video_id in tqdm(VTM): 
        video_id,pseudo_data = _pseudo_per_video(args,logger,video_id,VTM[video_id],step_task_occurrence,step2node,stat_logger)
        buffer[video_id] = pseudo_data
        if len(buffer) >= buffer_limit:
            flush_buffer_to_disk(buffer,save_path,logger)
            stat_logger.clear() 
        
    flush_buffer_to_disk(buffer,save_path,logger)
    return

