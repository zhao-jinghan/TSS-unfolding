import os
import numpy as np
from tqdm import tqdm
import random
import pickle

from datasets.build_knowledge.helper import *

def get_samples(args, logger):
    logger.info('getting video segment samples...')
    
    # get all video IDs
    video_id_path = os.path.join(args.howto100m_dir, 'video_id_list.pkl')
    video_id_list = pickle.load(open(video_id_path, 'rb'))
    
    samples = list()
    for v in tqdm(video_id_list):
        sim_scores_path = os.path.join(args.howto100m_dir, 'sim_scores', v + '.npy')
        sim_score = np.load(sim_scores_path)
        for c_idx in range(sim_score.shape[0]):
            samples.append((v, c_idx)) # (video_id,clip_idx)
    random.shuffle(samples)
    
    samples_id2name = dict()
    samples_name2id = dict()
    for i in range(len(samples)):
        samples_id2name[i] = samples[i]
        samples_name2id[samples[i]] = i
    
    os.makedirs(os.path.join(args.howto100m_dir, 'samples'), exist_ok=True)
    with open(os.path.join(args.howto100m_dir, 'samples/samples.pickle'), 'wb') as f:
        pickle.dump(samples_id2name, f)
    with open(os.path.join(args.howto100m_dir, 'samples/samples_reverse.pickle'), 'wb') as f:
        pickle.dump(samples_name2id, f)
    return 

def get_samples_feats_all(args,logger=None):
    sample_path = os.path.join(args.howto100m_dir, 'samples/samples.pickle')
    samples = pickle.load(open(sample_path, 'rb'))
    n = len(samples)
    
    feats_all = []
    
    video_feats = {}
    test = np.zeros((256,))
    
    for i in tqdm(range(n)):
        v, c_idx = samples[i] # c:clip idex
        
        if v not in video_feats:
            video_feat_path = os.path.join(args.howto100m_dir, 'BLIP features', f'{v}.npy')
            video_feat = np.load(video_feat_path)
            video_feats[v] = video_feat
        
        video_feat = video_feats[v].squeeze() # (segment_cnt,1,256)
        clip = video_feat[c_idx*3:c_idx*3+3]
        clip = np.mean(clip, axis=0)
     
        if clip.shape != test.shape:
            clip = test
        
        feats_all.append(clip) 
    
    feats_all = np.array(feats_all)
    print(feats_all.shape) 
    save_path = os.path.join(args.howto100m_dir, 'samples_feats_all.npy')
    np.save(save_path, feats_all) 

def load_samples(args,logger):
    if not os.path.exists(os.path.join(args.howto100m_dir, 'samples/samples.pickle')):
        from datasets.build_knowledge.get_samples import get_samples
        get_samples(args, logger)
    with open(os.path.join(args.howto100m_dir, 'samples/samples.pickle'), 'rb') as f:
        samples = pickle.load(f)
    with open(os.path.join(args.howto100m_dir, 'samples/samples_reverse.pickle'), 'rb') as f:
        samples_reverse = pickle.load(f)

    return samples, samples_reverse

import yaml
from utils.common_utils import Bunch

def load_samples_new(args,logger,target_stage):
    CoP_graph_config = getattr(args,f'graph_{target_stage}_config')
    with open(CoP_graph_config, 'rb') as f:
        config = yaml.safe_load(f)
        config = Bunch(config)
        
        N_unit = config.N_unit # int(config.t_video_shot // config.t_video_unit )
        print(N_unit,'-----N_unit')
    # feat dir 
    
    sample_dir = os.path.join(args.dataset_dir['CoP'], f'{config.visual_encoder}_{N_unit}_{target_stage}_level', 'pretrain_sample')
    if args.adapter_name == 'adapter_tx':
        sample_dir = os.path.join(args.dataset_dir['CoP'], f'{config.visual_encoder}_{N_unit}_{target_stage}_level_nomean', 'pretrain_sample')
    
    samples_id2name = pickle.load(open(os.path.join(sample_dir, 'samples.pkl'), 'rb'))
    samples_name2id = pickle.load(open(os.path.join(sample_dir, 'samples_reverse.pkl'), 'rb'))
    feats_all = pickle.load(open(os.path.join(sample_dir, 'feats_all.pkl'), 'rb'))
    
    print(len(samples_id2name))
    print(len(feats_all))
    return samples_id2name,feats_all
    
def load_VNM_by_stage(config,logger,target_stage):
    if target_stage == 'task':
        
        from datasets.build_CoP.task_level.pseudo_label_taskVNM import load_taskVNM_labels,load_taskVNM_GT
        VNM = load_taskVNM_GT(config,logger)
    elif target_stage == 'step':
        
        from datasets.build_CoP.step_level.pseudo_label_stepVNM import load_stepVNM_labels
        VNM = load_stepVNM_labels(config,logger)
     
    elif target_stage == 'state':  
        
        from datasets.build_CoP.state_level.pseudo_label_stateVNM import load_stateVNM_labels
        VNM = load_stateVNM_labels(config,logger)
    else :
        raise ValueError('target_stage must be in [task,step,state]')
    return VNM
  
def generate_samples_and_featsAll(args,logger,target_stage,config,check_valid=True):
    sample_dir = os.path.join(args.dataset_dir['CoP'], f'{config.visual_encoder}_{target_stage}_level', 'pretrain_sample')
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(os.path.join(sample_dir, 'feats_all'),exist_ok=True)
    feat_dir = os.path.join(args.dataset_dir['ht100m'],f'{config.visual_encoder}_video')
    samples = list()
    feats_all = {}

    VNM = load_VNM_by_stage(config,logger,target_stage)
    for video_id in tqdm(VNM):
        
        feat_path = os.path.join(feat_dir, video_id + '.npy')
        video_feat = np.load(feat_path)
        feats = np.mean(video_feat,axis=1)
        for i in range(feats.shape[0]):
            if len(VNM[video_id][i]['idx']) > 0 :
                samples.append((video_id, i))
      
        np.save(os.path.join(sample_dir, 'feats_all', video_id+'.npy'), feats)
        feats_all[video_id] = os.path.join(sample_dir, 'feats_all', video_id+'.npy')
    random.shuffle(samples)
    
    samples_id2name = dict()
    samples_name2id = dict()
    for i in range(len(samples)):
        samples_id2name[i] = samples[i]
        samples_name2id[samples[i]] = i
    
    pickle.dump(samples_id2name, open(os.path.join(sample_dir, 'samples.pkl'), 'wb'))
    pickle.dump(samples_name2id, open(os.path.join(sample_dir, 'samples_reverse.pkl'), 'wb'))
    pickle.dump(feats_all, open(os.path.join(sample_dir, 'feats_all.pkl'), 'wb'))
    return 


    