import os
import numpy as np
import time
import pickle
import yaml
from collections import defaultdict
from itertools import repeat
from multiprocessing import Pool
from scipy.sparse import csr_matrix
from utils.common_utils import *
from datasets.build_knowledge.helper import *
def generate_node(args, logger,wiki_step_feats):
    num_steps = len(wiki_step_feats)
    if args.cluster_sim:
        start_time = time.time()

        assert wiki_step_feats is not None
        logger.info(f'Start to cluster steps, step shape: {wiki_step_feats.shape}')

        wiki_step_feats = wiki_step_feats / np.linalg.norm(wiki_step_feats, axis=1, keepdims=True)

        from sklearn.cluster import AgglomerativeClustering
        config = args.cluster_config 
        clustering = AgglomerativeClustering(
            n_clusters=None, 
            linkage=config['linkage'], 
            distance_threshold=config['distance_threshold'],  # config['distance_threshold']
            metric=config['metric']).fit(wiki_step_feats)
        clustering = clustering[0] if type(clustering) == tuple else clustering
        n_cluster = clustering.n_clusters_  
        
        node2step, step2node = defaultdict(), defaultdict()
        for cluster_id in range(n_cluster):
            cluster_members = np.where(clustering.labels_ == cluster_id)[0]
            node2step[cluster_id] = cluster_members
            for step_id in cluster_members:
                step2node[step_id] = cluster_id

        logger.info(f'Nodes generate!!! {num_steps} -> {n_cluster}, spend {round(time.time()-start_time, 2)} s')
    else:
        n_cluster = num_steps
        
        node2step = {i: [i] for i in range(n_cluster)}
        step2node = {i: i for i in range(n_cluster)}
        
    # save 
    path_node2step = os.path.join(args.pwd,'node_label','node2step.pkl')
    path_step2node = os.path.join(args.pwd,'node_label','step2node.pkl')
    pickle.dump(node2step, open(path_node2step, 'wb'))
    pickle.dump(step2node, open(path_step2node, 'wb'))

    with open(args.cfg, "r") as file:
        data = yaml.safe_load(file) or {} 
        data['num_nodes'] = n_cluster
        data['num_steps'] = num_steps
    with open(args.cfg, "w") as file:
        yaml.dump(data, file, sort_keys=False)

    args.num_nodes = n_cluster
    args.num_steps = num_steps
    return 


def get_nodes(args, logger):
    os.makedirs(args.pwd, exist_ok=True)
    os.makedirs(args.pwd+'/step_label', exist_ok=True)
    os.makedirs(args.pwd+'/node_label', exist_ok=True)

    wiki_step_feats = load_npy(os.path.join(args.dataset_dir['wikihow'], f'MPNet_{args.step_text}.npy'))
    generate_node(args, logger, wiki_step_feats)
    

def load_nodes(args,logger):
    path_node2step = os.path.join(args.pwd,'node_label','node2step.pkl')
    path_step2node = os.path.join(args.pwd,'node_label','step2node.pkl')
    
    node2step = pickle.load(open(path_node2step,'rb'))
    step2node = pickle.load(open(path_step2node,'rb'))
    return node2step, step2node


def get_edges_from_wikihow(args,logger):
    graph = np.zeros((args.num_steps,args.num_steps))
    task2step = pickle.load(open(os.path.join(args.dataset_dir['wikihow'],'task2step.pkl'),'rb'))
    # step_id = (task_id,article_step_id) 

    for task_id in task2step:
        step_array = task2step[task_id]

        for i in range(len(step_array)-1):
            edge_from, edge_to = step_array[i],step_array[i+1]
            graph[edge_from][edge_to] += 1
    return graph

def get_edges_from_ht100m(args,sim_path):
    sim_score = np.load(sim_path)
    num_shots = sim_score.shape[0]
    config = args.graph_ht100m_config # by threshold:10
    edges_meta = [] 
    
    for i in range(num_shots-1): 
        edge_from_list,_ = find_matching_of_a_segment(
            sim_score[i],
            criteria=config['criteria'], 
            threshold=config['threshold'],
            topK=config['topK']
        )
        edge_to_list,_ = find_matching_of_a_segment(
            sim_score[i+1],
            criteria=config['criteria'], 
            threshold=config['threshold'],
            topK=config['topK']
        )
        
        for edge_from in edge_from_list: 
            for edge_to in edge_to_list: 
                if edge_from == edge_to:
                    continue
                edges_meta.append([edge_from,edge_to, 
                                sim_score[i][edge_from]*sim_score[i+1][edge_to]])
    return edges_meta

def threshold_and_normalize(args, logger, G, edge_min_aggconf=1000):
    G_new = np.zeros((G.shape[0], G.shape[0]))
    for i in range(G.shape[0]):
        for j in range(G.shape[0]):
            if G[i, j] > edge_min_aggconf:
                G_new[i, j] = G[i, j]
    G = G_new
    
    G_flat = G.reshape(G.shape[0]*G.shape[0],)
    x = [np.log(val) for val in G_flat if val != 0]
    assert len(x) > 0, 'No edges remain after thresholding! Please use a smaller edge_min_aggconf!'
    max_val, min_val = np.max(x), 0
    
    logger.info('normalizing edges...')
    G_new = np.zeros((G.shape[0], G.shape[0]))
    for i in range(G.shape[0]):
        for j in range(G.shape[0]):
            if G[i, j] > 0:
                G_new[i, j] = (np.log(G[i, j])-0)/(max_val-0)  # log min max norm
    G = G_new    
    return G

def get_node_transition_candidates(args, logger, step2node, G_wikihow, G_howto100m):
    candidates = defaultdict(list)
    
    for step_id in tqdm(range(len(step2node))):
        for direct_outstep_id in G_wikihow[step_id].indices:
            conf = G_wikihow[step_id, direct_outstep_id]
            
            node_id = step2node[step_id]
            direct_outnode_id = step2node[direct_outstep_id]
            
            candidates[(node_id, direct_outnode_id)].append(conf)

    for step_id in tqdm(range(len(step2node))):
        for direct_outstep_id in G_howto100m[step_id].indices:
            conf = G_howto100m[step_id, direct_outstep_id]
            
            node_id = step2node[step_id]
            direct_outnode_id = step2node[direct_outstep_id]
            
            candidates[(node_id, direct_outnode_id)].append(conf)
            
    return candidates


def keep_highest_conf_for_each_candidate(args, logger, candidates):
    edges = defaultdict()
    for (node_id, direct_outnode_id) in tqdm(candidates):
        max_conf = np.max(candidates[(node_id, direct_outnode_id)])
        
        edges[(node_id, direct_outnode_id)] = max_conf
    return edges


def build_pkg_adj_matrix(edges, num_nodes):
    pkg = np.zeros((num_nodes, num_nodes))
    for (node_id, direct_outnode_id) in tqdm(edges):
        pkg[node_id, direct_outnode_id] = edges[(node_id, direct_outnode_id)]
    return pkg
def unpack_param(params):
    args, sim_path = params
    return get_edges_from_ht100m(args, sim_path)
def get_edges(args,logger,sim_dir):
    logger.info('graph wikihow.....')
    G_wikihow = get_edges_from_wikihow(args,logger)
    logger.info('graph ht100m .....')
    sim_path_list = [os.path.join(sim_dir,file) for file in os.listdir(sim_dir) if file.endswith('.npy')]
    with Pool(processes=args.num_workers) as pool:
        edges_metas = list(tqdm(pool.imap(unpack_param, zip(repeat(args), sim_path_list)),
                                total=len(sim_path_list)))

    G_ht100m = np.zeros_like(G_wikihow) # (N_steps,N_steps)
    for edges_meta in edges_metas:
        for [edge_from, edge_to, confidence] in edges_meta:
            G_ht100m[edge_from, edge_to] += confidence
    G_ht100m = threshold_and_normalize(args, logger, G_ht100m, args.graph_ht100m_config['min_conf'])

    G_wikihow_csr, G_ht100m_csr = csr_matrix(G_wikihow), csr_matrix(G_ht100m)

    node2step, step2node = load_nodes(args, logger)

    node_transition_candidates = get_node_transition_candidates(
        args, logger, step2node, G_wikihow_csr, G_ht100m_csr)

    pkg_edges = keep_highest_conf_for_each_candidate(
        args, logger, node_transition_candidates)


    

def load_edges(args,logger): 
    edge_dir = os.path.join(args.pwd,'pseudo')
    pkg = np.load(os.path.join(edge_dir,'pkg.npy'))
    # G_wikihow_csr = pickle.load(open(os.path.join(edge_dir,'G_wikihow.pkl'), 'rb'))
    # G_ht100m_csr = pickle.load(open(os.path.join(edge_dir,'G_ht100m.pkl'), 'rb'))
    return pkg # G_wikihow_csr, G_ht100m_csr
