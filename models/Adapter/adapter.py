import torch.nn as nn
import torch
from models.misc import build_mlp


class Adapter(nn.Module):
    def __init__(self, args, logger,prediction=True):
        super(Adapter, self).__init__()
        
        self.args, self.logger = args, logger
        
        assert args.adapter_refined_feat_dim == args.s3d_hidden_dim 
        
        adapter_layers = [
            nn.Linear(args.s3d_hidden_dim, args.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.bottleneck_dim, args.adapter_refined_feat_dim),
        ]
        
        self.adapter = nn.Sequential(*adapter_layers)
      

        if not prediction:
            return 
        
        output_dim = {
            'taskVNM':args.task_num_nodes,

            'stepVNM':args.step_num_nodes,
            'stepTCL':args.step_num_nodes,
            'stepNRL':args.step_num_nodes,

            'stateVNM':args.state_num_nodes,
        }
        if args.pretrain_knowledge == 'CoP':
            for i,obj in enumerate(args.adapter_objective):
                
                answer_head = build_mlp(
                    input_dim=args.adapter_refined_feat_dim,
                    hidden_dims=[
                        output_dim[obj]//4,
                        output_dim[obj]//2,
                    ],
                    output_dim=output_dim[obj],
                ) 
                self.add_module(f'answer_head_{obj}', answer_head)
               
    def forward(self, segment_feat,nothing=None, prediction=True):
        refined_segment_feat = self.adapter(segment_feat) 
        if self.use_res:
            refined_segment_feat = self.alpha * refined_segment_feat + (1-self.alpha) * segment_feat # alpha 
        
        
        if not prediction: # 在downstream阶段
            return refined_segment_feat
        
        if self.args.pretrain_knowledge == 'CoP':
            outputs = []
            for obj in self.args.adapter_objective:
                cls_head = getattr(self, 'answer_head_'+obj)
                if type(cls_head) == list:
                    output = [cls_head[i](refined_segment_feat) for i in range(len(cls_head))]
                else:
                    output = cls_head(refined_segment_feat)
                outputs.append(output)
            return tuple(outputs)
        