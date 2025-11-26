import os 
import yaml
import numpy as np
from tqdm import tqdm
import pickle
import shutil
import time

def __sim_matrix__(A,B):
    A_norm = np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = np.linalg.norm(B, axis=1, keepdims=True)

    dot_product = np.dot(A, B.T)
    cosine_similarity_matrix = dot_product #/ (A_norm * B_norm.T)
     
    return cosine_similarity_matrix
def get_sim(args,logger,simscore_dir,path_text_feat,check_ht100_range=True):
    start_time = time.time()
    logger.info('calc simscore ...'+simscore_dir)

    os.makedirs(simscore_dir,exist_ok=True)
    text_feature = np.load(path_text_feat)
    
    extra_dim = None

    video_feat_dir = os.path.join(args.dataset_dir['ht100m'],f'{args.visual_encoder}_video')


    for filename in tqdm(os.listdir(video_feat_dir)): 
        if not filename.endswith('.npy'):
            continue
        video_id = filename.split('.')[0]
        
        video_path = os.path.join(video_feat_dir, filename)
        video_feat = np.load(video_path) 
        
        shot_feat = np.mean(video_feat,axis=1)
        assert shot_feat.shape[-1] == 512 and len(shot_feat.shape) == 2 ,print(shot_feat.shape)
        
        sim_scores = __sim_matrix__(shot_feat,text_feature) # (shot_cnt,text_cnt)
        
        if extra_dim is not None:
            sim_scores = sim_scores.reshape(sim_scores.shape[0],extra_dim,-1) # (shot_cnt,extra_dim=3,text_cnt)

        sim_scores_path = os.path.join(simscore_dir, video_id + '.npy')
        np.save(sim_scores_path,sim_scores)
        
        if filename != video_id + '.npy':
            os.rename(os.path.join(video_feat_dir, filename), os.path.join(video_feat_dir, video_id + '.npy'))

    
