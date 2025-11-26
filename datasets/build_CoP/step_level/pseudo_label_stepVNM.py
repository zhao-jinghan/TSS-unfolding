import os
import numpy as np
import pickle
import multiprocessing
from utils.common_utils import *
from datasets.build_knowledge.helper import *
from datasets.build_CoP.step_level.graph_utils import *


def _save_path(args,logger=None): 
    # return : save_path(str)
    pseudo_dir = os.path.join(args.pwd,'pseudo')
    os.makedirs(pseudo_dir,exist_ok=True)
    
    config = args.stepVNM_config
    param = None
    if config['criteria'] == 'threshold':
        param = str(config['threshold'])
    elif config['criteria'] == 'topK':
        param = str(config['topK'])
    elif config['criteria'] == 'threshold+topK':
        param = str(config['threshold'])+'_'+str(config['topK'])
    filename = f"stepVNM_{config['criteria']}_{param}.pkl" 
    
    save_path = os.path.join(pseudo_dir,filename)
    return save_path

def load_stepVNM_labels(args,logger):
    save_path = _save_path(args,logger)

    stepVNM_pseudo_labels = pickle.load(open(save_path,'rb'))
    return stepVNM_pseudo_labels

def _pseudo_per_shot(args,logger,simscore,step2node,prune_filter=None):
    node_scores = np.zeros(args.num_nodes,dtype=np.float32) 
    config = args.stepVNM_config
    
    # init filter 
    mask = simscore > config['threshold'] if 'threshold' in config['criteria'] else simscore > 0
    
    for step_id, mask_i in enumerate(mask):
        if not mask_i:
            continue
        node_id = step2node[step_id]
        
        node_scores[node_id] = max(node_scores[node_id], simscore[step_id])

    idx,scores = find_matching_of_a_segment(
        node_scores, 
        criteria=config['criteria'], 
        threshold=config['threshold'],
        topK=config['topK'])
    
    return idx,scores

def _pseudo_per_video(args,logger,video_id,sim_dir,step2node,pseudo_label,lock,stat_logger=None): 
    sim_score = np.load(os.path.join(sim_dir,f'{video_id}.npy'))
    num_shots = sim_score.shape[0]

    _pseudo_video = []
    for i in range(num_shots):
        idx,scores = _pseudo_per_shot(args,logger,sim_score[i],step2node)

        _pseudo_video.append({
            'idx':idx,
            'scores':scores
        })

        if stat_logger is not None:
            stat_logger.update('stepVNM_cnt',len(idx))
            stat_logger.update('stepVNM_score',scores)
    
    with lock: 
        pseudo_label[video_id] = _pseudo_video
        if (len(pseudo_label) + 1) % 10000 == 0:
            pseudo_label_plain = dict(pseudo_label)  
            save_path = _save_path(args,logger)
            if os.path.exists(save_path):
                history_part = pickle.load(open(save_path,'rb'))
                history_part.update(pseudo_label_plain)

                # save 
                pickle.dump(history_part,open(save_path,'wb'))
                logger.info(f'save pseudo [part] : {len(history_part)}')
            else:  
                pickle.dump(pseudo_label_plain,open(save_path,'wb'))
                logger.info(f'save pseudo [part] : {len(pseudo_label_plain)}')
                
            pseudo_label.clear() 

def unpack_param(param_tuple): 
    return _pseudo_per_video(*param_tuple)

def pseudo_label_stepVNM(args,logger,sim_dir):
    save_path = _save_path(args,logger)
    assert not os.path.exists(save_path), "pseudo label already exists"
    path_step2node = os.path.join(args.pwd,'node_label','step2node.pkl')
    step2node  = pickle.load(open(path_step2node, 'rb'))
    

    with multiprocessing.Manager() as manager: # shared dict
        pseudo_label = manager.dict()
        lock = manager.Lock()

        params = [] 
        for filename in os.listdir(sim_dir):
            if not filename.endswith('.npy'):
                continue
            video_id = filename.split('.')[0]
            params.append((args,logger,video_id,sim_dir,step2node,pseudo_label,lock)) 
            
        
        with multiprocessing.Pool(processes=32) as pool:
             for _ in tqdm(pool.imap_unordered(unpack_param, params), total=len(params)):
                pass  #
        pseudo_label_plain = dict(pseudo_label)
        save_path = _save_path(args,logger)

        if os.path.exists(save_path):
            history_part = pickle.load(open(save_path,'rb'))
            history_part.update(pseudo_label_plain)

            pickle.dump(history_part,open(save_path,'wb'))
        else:  
            pickle.dump(pseudo_label_plain,open(save_path,'wb'))
                
    return pseudo_label
        

