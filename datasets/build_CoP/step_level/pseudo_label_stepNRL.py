import os
import numpy as np
import pickle
from multiprocessing import Pool
from datasets.build_CoP.step_level.pseudo_label_stepVNM import load_stepVNM_labels
from datasets.build_CoP.step_level.get_step_nodes import load_edges
from utils.common_utils import *
from datasets.build_knowledge.helper import *
from datasets.build_knowledge.get_samples import load_samples

def _save_path(args,logger=None,hop_i=None): 
    pseudo_dir = os.path.join(args.pwd,'pseudo')
    os.makedirs(pseudo_dir,exist_ok=True)
    config = args.stepNRL_config
    param = None
    if config['criteria'] == 'threshold':
        param = str(config['threshold'])
    elif config['criteria'] == 'topK':
        param = str(config['topK'])
    elif config['criteria'] == 'threshold+topK':
        param = str(config['threshold'])+'_'+str(config['topK'])
    filename = f"stepNRL_{config['criteria']}_{param}_{config['khop']}.pkl" if hop_i is None \
                else f"stepNRL_{config['criteria']}_{param}_{config['khop']}_i{hop_i}.pkl"
    save_path = os.path.join(pseudo_dir,filename)
    return save_path
def _pseudo_per_shot(args,logger,source_nodes, source_conf, graph):
    num_nodes = graph.shape[0]
    NRL_candidate = np.zeros(num_nodes)
    for start_node, start_score in zip(source_nodes, source_conf):
        edges = graph[start_node] > 0
        for neighbor in np.where(edges)[0]:
            weight = graph[start_node, neighbor]  
            next_score = start_score * weight 
            NRL_candidate[neighbor] = max(next_score,NRL_candidate[neighbor]) 

    config = args.stepNRL_config
    idx, scores = find_matching_of_a_segment(
            NRL_candidate,
            criteria=config['criteria'], 
            threshold=config['threshold'],
            topK=config['topK']
        )
    return {
        'idx':idx,
        'scores':scores
    }

def _pseudo_per_video(args,logger,video_id,
                      VNM_pseudo_labels,graph,graph_revert,hop_i,
                      NRL_hop_last,stat_logger=None): 
    num_shots = len(VNM_pseudo_labels)
    _pseudo_video = []
    for i in range(num_shots):
        if hop_i == 1:
            source_nodes,source_confs = VNM_pseudo_labels[i]['idx'],VNM_pseudo_labels[i]['scores']

            shot_NRL_revert = _pseudo_per_shot(args,logger,source_nodes,source_confs,graph_revert)
            shot_NRL = _pseudo_per_shot(args,logger,source_nodes,source_confs,graph)
        else: 
            last_hop_revert,last_hop = NRL_hop_last[video_id][i]
            source_nodes,source_confs = last_hop_revert['idx'],last_hop_revert['scores']

            shot_NRL_revert = _pseudo_per_shot(args,logger,source_nodes,source_confs,graph_revert)
            shot_NRL = _pseudo_per_shot(args,logger,source_nodes,source_confs,graph)

        _pseudo_video.append((shot_NRL_revert,shot_NRL))
        
        if stat_logger is not None:
            stat_logger.update('stepNRL_cnt',len(shot_NRL['idx']))
            stat_logger.update('stepNRL_score',shot_NRL['scores'])
    
    return video_id,_pseudo_video
def unpack_param(param_tuple): 
    return _pseudo_per_video(*param_tuple)

def load_stepNRL_labels(args,logger):
    save_path = _save_path(args,logger)
    return pickle.load(open(save_path,'rb')) 

def pseudo_label_stepNRL(args,logger):
    VNM = load_stepVNM_labels(args,logger)
    graph = load_edges(args,logger) 
    graph_revert = np.transpose(graph) # 反向

    
    for hop in range(1,args.stepNRL_config['khop']+1):
        logger.info(f"generate NRL khop={hop} pseudo labels...")
           
        save_path = _save_path(args,logger,hop_i=hop)
        assert not os.path.exists(save_path), "pseudo label already exists"
        
        if hop == 1: 
            NRL_hop_last = [] 
        else: 
            NRL_hop_last = pickle.load(open(_save_path(args,logger,hop_i=hop-1),'rb'))

        buffer = {}
        buffer_limit = 5000  
        for video_id in tqdm(VNM): 
            video_id, pseudo_data = _pseudo_per_video(args,logger,video_id,
                                        VNM[video_id],graph,graph_revert,hop,
                                        NRL_hop_last)
            buffer[video_id] = pseudo_data
            if len(buffer) >= buffer_limit:
                flush_buffer_to_disk(buffer,save_path,logger)
        flush_buffer_to_disk(buffer,save_path,logger)

    all_NRL = []

    for hop in range(1,args.stepNRL_config['khop']+1):
        NRL_i = pickle.load(open(_save_path(args,logger,hop_i=hop),'rb'))
        all_NRL.append(NRL_i)
    
    pickle.dump(all_NRL,open(_save_path(args,logger),'wb'))
    
    return all_NRL

