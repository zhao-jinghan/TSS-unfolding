import os
import numpy as np
import time
import pickle
import yaml
from collections import defaultdict
from utils.common_utils import *
from datasets.build_knowledge.helper import *

def cluster(feature,config):
    from sklearn.cluster import AgglomerativeClustering
    clustering = AgglomerativeClustering(
          n_clusters=None, 
          linkage=config['linkage'], 
          distance_threshold=config['distance_threshold'], 
          metric=config['metric']).fit(feature)
    n_cluster = clustering.n_clusters_

    nodes = []
    for cluster_id in range(n_cluster):
        cluster_members = np.where(clustering.labels_ == cluster_id)[0] 
        nodes.append(cluster_members)
    
    return nodes 
def generate_node(args, logger,wiki_state_feats):
    num_steps = wiki_state_feats.shape[1]
    num_states = num_steps * 3
    if args.cluster_sim:
        start_time = time.time()
        logger.info('Start clustering...')
        B_state, T_state, A_state = wiki_state_feats[0], wiki_state_feats[1], wiki_state_feats[2]
        
        assert wiki_state_feats is not None

        def generate_mapping(cluster_result,node2state,state2node,state_type):
            STATE_ID_OFFSET = state_type * num_steps

            for node in cluster_result:
                node_id = len(node2state) 
                node2state[node_id] = []
                for state_id in node:
                    state2node[state_id+STATE_ID_OFFSET] = node_id
                    node2state[node_id].append(state_id+STATE_ID_OFFSET)
            return len(cluster_result)

        node2state = defaultdict()
        state2node = defaultdict()

        B_nodes = cluster(B_state,config=args.B_state_cluster_config) 
        num_B_nodes = generate_mapping(B_nodes,node2state,state2node,state_type=0)

        T_nodes = cluster(T_state,config=args.T_state_cluster_config) 
        num_T_nodes = generate_mapping(T_nodes,node2state,state2node,state_type=1)

        A_nodes = cluster(A_state,config=args.A_state_cluster_config)
        num_A_nodes = generate_mapping(A_nodes,node2state,state2node,state_type=2)

        num_nodes = num_B_nodes + num_T_nodes + num_A_nodes
    else:
        num_nodes = num_states
        assert num_states % 3 == 0 

        num_B_nodes,num_T_nodes,num_A_nodes = num_steps,num_steps,num_steps

        node2state = {i:[i] for i in range(num_states)}
        state2node = {i:i for i in range(num_states)}

        
    path_node2state = os.path.join(args.pwd,'node_label','node2state.pkl')
    path_state2node = os.path.join(args.pwd,'node_label','state2node.pkl')
    pickle.dump(node2state, open(path_node2state, 'wb'))
    pickle.dump(state2node, open(path_state2node, 'wb'))

    with open(args.cfg, "r") as file:
        data = yaml.safe_load(file) or {} 
        data['num_nodes'] = num_nodes
        data['num_states'] = num_states
        data['num_B_nodes'] = num_B_nodes
        data['num_T_nodes'] = num_T_nodes
        data['num_A_nodes'] = num_A_nodes

    with open(args.cfg, "w") as file:
        yaml.dump(data, file, sort_keys=False)

    args.num_nodes = num_nodes
    args.num_states = num_states

    args.num_B_nodes = num_B_nodes
    args.num_T_nodes = num_T_nodes
    args.num_A_nodes = num_A_nodes
    return 


def get_nodes(args, logger):
    os.makedirs(args.pwd, exist_ok=True)
    os.makedirs(args.pwd+'/state_label', exist_ok=True)
    os.makedirs(args.pwd+'/node_label', exist_ok=True)

    wiki_state_feats = load_npy(os.path.join(args.dataset_dir['wikihow'], f'MPNet_{args.state_text}.npy')) # shape (3, num_steps,text_dim)

    generate_node(args, logger, wiki_state_feats)


def load_nodes(args,logger,state_type=None):
    path_node2state = os.path.join(args.pwd,'node_label','node2state.pkl')
    path_state2node = os.path.join(args.pwd,'node_label','state2node.pkl')
    
    node2state = pickle.load(open(path_node2state,'rb'))
    state2node = pickle.load(open(path_state2node,'rb'))
    
    return node2state, state2node

