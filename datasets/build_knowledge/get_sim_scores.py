import os
import numpy as np
from tqdm import tqdm
import time

from datasets.build_knowledge.helper import *


def gatther_all_frame_S3D_embeddings(args, logger):
    start_time = time.time()
    
    # get all video IDs
    videos = get_all_video_ids(args, logger)
    
    frame_embeddings = []
    frame_lookup_table = []
    
    videos_missing_features = set()
    for v in tqdm(videos):
        try:
            video_s3d = np.load(os.path.join(args.howto100m_dir, 'feats', v, 'video.npy'))
            # video_s3d shape: (num_clips, num_subclips, 512)
            
            for c_idx in range(video_s3d.shape[0]): 
                frame_embeddings.append(np.float64(np.mean(video_s3d[c_idx], axis=0)))
                # shape:(num_clips,512)
                frame_lookup_table.append((v, c_idx))
                # mean embedding of each clip 
                

        except FileNotFoundError:
            videos_missing_features.add(v)

    logger.info("number of videos missing visual S3D features: {}".format(
        len(videos_missing_features)))
    if len(videos_missing_features) > 0:
        with open('videos_missing_features.pickle', 'wb') as f:
            pickle.dump(videos_missing_features, f)
    assert len(videos_missing_features) == 0, ("There are videos missing features! "
                                               + "Please check saved videos_missing_features.pickle.")
   
    frame_embeddings = np.array(frame_embeddings)
    
    logger.info("segment frame embeddings shape: {}".format(frame_embeddings.shape))
    # segment video embeddings shape: (3741608, 512) for the subset
    # segment video embeddings shape: (51053844, 512) for the fullset
    logger.info("getting all segment frame embeddings took {} s".format(round(time.time()-start_time, 2)))
    return frame_embeddings, frame_lookup_table


'''
def find_step_similarities_for_segments_using_frame(
    args, logger,
    step_des_feats, segment_video_embeddings, segment_video_lookup_table):
    
    start = time.time()
           
    for segment_id in tqdm(range(len(segment_video_embeddings))):
        v, cidx = segment_video_lookup_table[segment_id]
        save_path = os.path.join(args.howto100m_dir, 'sim_scores', v, 'segment_{}.npy'.format(cidx))
        if not os.path.exists(save_path):
            
            # dot product as similarity score
            sim_scores = np.einsum('ij,ij->i',
                                   step_des_feats, 
                                   segment_video_embeddings[segment_id][np.newaxis, ...])
            
            os.makedirs(os.path.join(args.howto100m_dir, 'sim_scores', v), exist_ok=True)
            np.save(save_path, sim_scores)
            
    logger.info('finding step similarity scores for segments using frames ' + 
                'took {} seconds'.format(time.time() - start))
    # os._exit(0)
    return 
'''

# def get_sim_scores(args, logger):
#     step_des_feats = get_step_des_feats(args, logger, language_model="S3D")
        
#     segment_video_embeddings, segment_video_lookup_table = \
#         gatther_all_frame_S3D_embeddings(args, logger)

#     find_step_similarities_for_segments_using_frame(
#             args, logger, 
#             step_des_feats, segment_video_embeddings, segment_video_lookup_table)
    

def get_sim_scores(args, logger):
    logger.info('calc sim scores...')
    os.makedirs(os.path.join(args.howto100m_dir, 'sim_scores_text'), exist_ok=True)

    headline_feat_path = os.path.join(args.wikihow_dir, 'text_feats.npy')
    headline_feat = np.load(headline_feat_path)
    video_feat_dir = os.path.join(args.howto100m_dir, args.VL_encoder+' features')

    video_id_list = []
    for filename in tqdm(os.listdir(video_feat_dir)):
        if filename.endswith('.npy'):
            video_path = os.path.join(video_feat_dir, filename)
            video_id = filename.split('.')[0]
            video_id_list.append(video_id)

            sim_scores_path = os.path.join(args.howto100m_dir, 'sim_scores_text', video_id + '.npy')

            if os.path.exists(sim_scores_path):
                continue
            
            video_feat = np.load(video_path) # (duration/3.2,256)
            N_segment = 3 # 3.2s *3 = 9.6s 
            segment_feat = []
            for i in range(0, video_feat.shape[0],N_segment):
                group = video_feat[i:i+N_segment]
                segment_feat.append(np.mean(group,axis=0)) # 3个feat取平均
            
            segment_feat = np.array(segment_feat) # n个32帧
            
            
            
            sim_scores = np.einsum('ij,kj->ik', segment_feat,headline_feat) # (STEP_CNT,SEGMENT_CNT)
            np.save(sim_scores_path,sim_scores)
            
            if filename != video_id + '.npy':
                os.rename(os.path.join(video_feat_dir, filename), os.path.join(video_feat_dir, video_id + '.npy'))
    
    pickle.dump(video_id_list, open(os.path.join(args.howto100m_dir, 'video_id_list.pkl'), 'wb'))

