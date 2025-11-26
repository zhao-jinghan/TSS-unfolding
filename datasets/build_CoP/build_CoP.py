import os 
import sys
sys.path.insert(0, os.path.abspath('./'))
from datasets.build_CoP.get_simscore import get_sim

def build_graph_task(args,logger):
     from datasets.build_CoP.task_level.get_task_nodes import get_nodes
     from datasets.build_CoP.task_level.pseudo_label_taskVNM import pseudo_label_taskVNM,real_label_taskVNM
     
     get_nodes(args,logger) 
     sim_dir = os.path.join(args.pwd,'task_label',f'sim--{args.visual_encoder}--{args.text_encoder}_{args.task_text}')
     get_sim(args,logger,
               simscore_dir=sim_dir,
               path_text_feat=os.path.join(args.pwd,'task_label', f'{args.task_source}--{args.text_encoder}--{args.task_text}.npy'), 
               ) 
     
     pseudo_label_taskVNM(args,logger,sim_dir)
     if 'ht100m' in args.task_source:
          real_label_taskVNM(args,logger,sim_dir)



def build_graph_step(args,logger):
     from datasets.build_CoP.step_level.get_step_nodes import get_nodes, get_edges,pruning
     from datasets.build_CoP.step_level.pseudo_label_stepVNM import pseudo_label_stepVNM
     from datasets.build_CoP.step_level.pseudo_label_stepTCL import pseudo_label_stepTCL
     from datasets.build_CoP.step_level.pseudo_label_stepNRL import pseudo_label_stepNRL
     logger.info('debug start')

     sim_dir = os.path.join(args.pwd,'step_label',f'sim--{args.visual_encoder}--{args.text_encoder}_{args.step_text}')
     get_nodes(args,logger) 
     get_sim(args,logger,
               simscore_dir=sim_dir,
               path_text_feat=os.path.join(args.dataset_dir['wikihow'], f'{args.text_encoder}_{args.step_text}.npy'),
               )
     
     pseudo_label_stepVNM(args,logger,sim_dir)
     get_edges(args,logger,sim_dir)
     pseudo_label_stepTCL(args,logger)
     pseudo_label_stepNRL(args,logger)

def build_graph_state(args,logger):
     from datasets.build_CoP.state_level.get_state_nodes import get_nodes
     from datasets.build_CoP.state_level.pseudo_label_stateVNM import pseudo_label_stateVNM
     
     sim_dir = os.path.join(args.pwd,'state_label',f'sim--{args.visual_encoder}--{args.text_encoder}_{args.state_text}')
     
     get_nodes(args,logger)
     get_sim(args,logger,
               simscore_dir=sim_dir,
               path_text_feat=os.path.join(args.dataset_dir['wikihow'], f'{args.text_encoder}_{args.state_text}.npy'),
               ) # simscore shape (3,num_steps,dim)
              
     pseudo_label_stateVNM(args,logger,sim_dir)


from utils.common_utils import set_seed,getLogger
from args.args_build_CoP import get_args_parser 

if __name__ == '__main__':
    args = get_args_parser()
    set_seed(args.seed)

    # set logger 
    logfile_path = os.path.abspath(
        os.path.join(args.log_dir, f'builCoP_debug.log'))
    logger = getLogger(name=__name__, path=logfile_path)

    if args.level == 'task':
         build_graph_task(args,logger)
    elif args.level == 'step':
         build_graph_step(args,logger)
    elif args.level == 'state':
         build_graph_state(args,logger)
    else:
         raise ValueError('level must be task or state')
    logger.info('start generate sample...')
  