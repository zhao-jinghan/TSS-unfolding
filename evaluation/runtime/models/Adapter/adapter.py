import torch.nn as nn
import torch
from models.misc import build_mlp

'''
check:
all module is trainable

关于BN,dropout这些参数的必要性
'''
class Adapter(nn.Module):
    def __init__(self, args, logger,prediction=True):
        super(Adapter, self).__init__()

        self.args, self.logger = args, logger

        assert args.adapter_refined_feat_dim == args.s3d_hidden_dim

        adapter_layers = [
            nn.Linear(args.s3d_hidden_dim, args.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.bottleneck_dim, args.adapter_refined_feat_dim),
            # tmt add
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(args.adapter_refined_feat_dim),
        ]

        self.adapter = nn.Sequential(*adapter_layers)
        if hasattr(args, "adapter_res_alpha") :
            self.use_res = True
            self.alpha = nn.Parameter(torch.tensor(args.adapter_res_alpha))  # 初始值为 0.5，可以调整为其他值
        else:
            self.use_res = False

        if not prediction:
            return

        cls_head_num = {
            'taskVNM':1,

            'stepVNM':1,
            'stepVTM':1,
            'stepTCL':1,
            'stepNRL':2*args.pretrain_stepNRL_khop,

            'stateVNM':1,
            'stateT2BA':2
        }
        output_dim = {
            'taskVNM':args.task_num_nodes,

            'stepVNM':args.step_num_nodes,
            'stepVTM':args.num_tasks,
            'stepTCL':args.step_num_nodes,
            'stepNRL':args.step_num_nodes,

            'stateVNM':args.state_num_nodes,
            'stateT2BA':args.state_num_nodes
        }
        if args.pretrain_knowledge == 'CoP':
            for i,obj in enumerate(args.adapter_objective):
                if cls_head_num[obj] == 1:
                    if obj == 'stepVTM':
                        answer_head = build_mlp(
                            input_dim=args.adapter_refined_feat_dim,
                            hidden_dims=[
                                output_dim[obj]//2,
                            ],
                            output_dim=output_dim[obj],
                        )
                        self.add_module(f'answer_head_{obj}', answer_head)
                    else:
                        answer_head = build_mlp(
                            input_dim=args.adapter_refined_feat_dim,
                            hidden_dims=[
                                output_dim[obj]//4,
                                output_dim[obj]//2,
                            ],
                            output_dim=output_dim[obj],
                        )
                        self.add_module(f'answer_head_{obj}', answer_head)
                else:
                    answer_head = [
                        build_mlp(
                        input_dim=args.adapter_refined_feat_dim,
                        hidden_dims=[
                            output_dim[obj]//4,
                            output_dim[obj]//2,
                        ],
                        output_dim=output_dim[obj]) for _ in range(cls_head_num[obj])
                    ]
                    for sub_idx,model in enumerate(answer_head):
                        self.add_module(f'answer_head_{obj}_{sub_idx}', model)

                    setattr(self, f'answer_head_{obj}', answer_head)
        if args.pretrain_knowledge == 'DS':
            self.answer_head = build_mlp(
                input_dim=args.adapter_refined_feat_dim,
                hidden_dims=[args.adapter_num_classes//6,
                             args.adapter_num_classes//4,
                             args.adapter_num_classes//2],
                output_dim=args.adapter_num_classes)
    def forward(self, segment_feat,nothing=None, prediction=True):
        """
        - segment_feat: (B, 512)
        """
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
        if self.args.pretrain_knowledge == 'DS':
            return self.answer_head(refined_segment_feat)
