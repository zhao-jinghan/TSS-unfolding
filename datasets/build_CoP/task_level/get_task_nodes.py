import os
import shutil
import numpy as np
import time
import pickle
import yaml
from collections import defaultdict
from utils.common_utils import *
from datasets.build_knowledge.helper import *

def merge_task_label(args,logger):
    # check intersection
    task_name_corpus = set()

    for dataset in args.task_source.split('_'):
        dataset_dir = args.dataset_dir[dataset]
        task_names = pickle.load(open(os.path.join(dataset_dir,'task_names.pkl'),'rb'))
        task_names = set(task_names)
        assert len(task_name_corpus.intersection(task_names)) == 0
        task_name_corpus.update(task_names)

    new_id2name = {}
    new_name2id = {}
    new_name_feat = []
    offset = len(new_id2name)
    
    for dataset in args.task_source.split('_'):
        dataset_dir = args.dataset_dir[dataset]
        id2name = pickle.load(open(os.path.join(dataset_dir,'task_id2name.pkl'),'rb'))
        
        new_id2name.update({i+offset: name for i, name in id2name.items()})
        new_name2id.update({name: i+offset for i, name in id2name.items()})
        offset = len(new_id2name)

        new_name_feat.append(np.load(os.path.join(dataset_dir,f'{args.text_encoder}_{args.task_text}.npy')))
    
    new_name_feat = np.concatenate(new_name_feat, axis=0)
    assert new_name_feat.shape[0] == len(new_id2name)

    logger.info(f'new_name_feature shape: {new_name_feat.shape}')
    # save 
    save_dir = os.path.join(args.pwd,'task_label')
    path_id2name = save_dir + f'/{args.task_source}--task_id2name.pkl'
    path_name2id = save_dir + f'/{args.task_source}--task_name2id.pkl'
    path_feature = save_dir + f'/{args.task_source}--{args.text_encoder}--{args.task_text}.npy'


    pickle.dump(new_id2name, open(path_id2name,'wb'))
    pickle.dump(new_name2id, open(path_name2id,'wb'))
    np.save(path_feature,new_name_feat)


def generate_node(args, logger):
    if False: #args.cluster_sim:
        start_time = time.time()

        path_MPNet_feature = os.path.join(args.pwd,'task_label',f'{args.task_source}--MPNet--{args.task_text}.npy')
        feature = np.load(path_MPNet_feature) 
        assert feature is not None

        from sklearn.cluster import AgglomerativeClustering
        config = args.cluster_config 
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            linkage=config['linkage'], 
            distance_threshold=config['distance_threshold'],
            metric=config['metric']).fit(feature) 
    
        n_cluster = clustering.n_clusters_
        
        node2task, task2node = defaultdict(), defaultdict()
        for cluster_id in range(n_cluster):
            cluster_members = np.where(clustering.labels_ == cluster_id)[0]
            node2task[cluster_id] = cluster_members
            for step_id in cluster_members:
                task2node[step_id] = cluster_id

    else:
        path_id2name = os.path.join(args.pwd,'task_label',f'{args.task_source}--task_id2name.pkl')
        n_cluster = len(pickle.load(open(path_id2name, 'rb')))
        
        node2task = {i: [i] for i in range(n_cluster)}
        task2node = {i: i for i in range(n_cluster)}
        
    path_node2task = os.path.join(args.pwd,'node_label','node2task.pkl')
    path_task2node = os.path.join(args.pwd,'node_label','task2node.pkl')
    pickle.dump(node2task, open(path_node2task, 'wb'))
    pickle.dump(task2node, open(path_task2node, 'wb'))

    with open(args.cfg, "r") as file:
        data = yaml.safe_load(file) or {} 
        data['num_nodes'] = n_cluster
        data['num_tasks'] = len(task2node)
    with open(args.cfg, "w") as file:
        yaml.dump(data, file, sort_keys=False)

    args.num_nodes = n_cluster
    args.num_tasks = len(task2node)
    return  #  node2task, task2node,n_cluster

def get_nodes(args,logger):
    os.makedirs(args.pwd, exist_ok=True)
    os.makedirs(args.pwd+'/task_label', exist_ok=True)
    os.makedirs(args.pwd+'/node_label', exist_ok=True)

    logger.info(f'merge task label for :{args.task_source}')
    merge_task_label(args,logger)
    
    generate_node(args,logger)
    

def load_nodes(args,logger):
    path_node2task = os.path.join(args.pwd,'node_label','node2task.pkl')
    path_task2node = os.path.join(args.pwd,'node_label','task2node.pkl')
    
    node2task = pickle.load(open(path_node2task,'rb'))
    task2node = pickle.load(open(path_task2node,'rb'))
    
    return node2task,task2node
