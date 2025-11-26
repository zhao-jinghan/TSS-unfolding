import os
import numpy as np 
# os.environ['OMP_NUM_THREADS'] = '1'
import sys
import pickle
sys.path.insert(0, os.path.abspath('./'))
import time
import platform
from utils.common_utils import calc_grad
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm 
from args.args_CoP_pretrain import get_args_parser
from datasets import return_dataset
from models import create_model
from utils.common_utils import (
    set_seed, 
    getLogger, need_logging, 
    save_checkpoint, trim,
    AverageMeter, global_meters_all_avg, accuracy, multilabel_cls_exact_match,
    get_rank, is_main_process)

import torch
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
torch.autograd.set_detect_anomaly(True)
import torch.distributed as dist
import csv 

def load_ckp(args,logger,adapter_model):
    adapter_checkpoint = torch.load(args.checkpoint, map_location=args.device)
    adapter_params = adapter_checkpoint['state_dict']
    adapter_params = trim(adapter_params)
    adapter_model.load_state_dict(
        adapter_params, strict=True)

def model_ddp_config(args,logger,model):
    if args.use_ddp:  # distributed training
        local_rank = int(os.environ['LOCAL_RANK'])
        if need_logging(args):
            logger.info('Rank: {} The local_rank is {}'.format(rank, local_rank))
        model.to(args.device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=False)
    else:
        model = model.to(args.device)
    adapter_model_n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if need_logging(args):
        logger.info('Rank: {} Number of params is {} for < adapter > training model'.format(rank, adapter_model_n_parameters))
        
    return model
    
def valid_mask(pred,gt):
    return pred,gt


def get_pseudo_label_criterion(args,logger,objective:list): 
    from torch.nn import BCEWithLogitsLoss as loss
    criterion_set = { 
        'taskVNM': loss().to(args.device) ,  

        'stepVNM': loss().to(args.device), 
        'stepTCL':  loss().to(args.device),
        'stepNRL': [loss().to(args.device) for _ in range(2*args.pretrain_stepNRL_khop)],
        
        'stateVNM': loss().to(args.device), 

    }
    return {obj:criterion_set[obj] for obj in objective}
def return_dataloader(args,logger,dataset,collate_fn=None):
    if args.use_ddp and not args.partition_dataset:  # distributed training
        adapter_train_sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset, shuffle=True)  
        
        dataloader = DataLoader(dataset,
                                batch_size=args.adapter_batch_size,
                                sampler=adapter_train_sampler,
                                num_workers=args.num_workers, 
                                collate_fn=collate_fn,
                                pin_memory=True)
    else:
        if args.use_ddp:
            from monai.data import ThreadDataLoader  # faster
            dataloader = ThreadDataLoader(dataset,
                                            batch_size=args.adapter_batch_size,
                                            shuffle=True,
                                            collate_fn=collate_fn,
                                            num_workers=0)
        else:
            dataloader = DataLoader(dataset,
                                    batch_size=args.adapter_batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers, 
                                    collate_fn=collate_fn,
                                    pin_memory=True)

    return dataloader
def main_train_adapter(args,logger):
    adapter_train_dataset = return_dataset(args, logger, 'HowTo100M')
    adapter_train_loader = return_dataloader(args,logger,adapter_train_dataset)
    
    adapter_model = create_model(args, logger, args.adapter_name)
    adapter_model = model_ddp_config(args,logger,adapter_model)

    if args.load_pretrained: # Load checkpoint
        if need_logging(args):
            logger.info('load ckp from...'+args.checkpoint)
        if args.use_ddp:
            load_ckp(args,logger,adapter_model.module.adapter)   
        else:
            load_ckp(args,logger,adapter_model.adapter)
    

    adapter_criterion = get_pseudo_label_criterion(args,logger,args.adapter_objective)
    assert len(adapter_criterion) > 0

    # Define adapter optimizer
    if args.adapter_optimizer == 'adam':
        adapter_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, adapter_model.parameters()), 
            lr=args.adapter_learning_rate, weight_decay=args.adapter_weight_decay)
    else:
        if need_logging(args):
            logger.info('Rank: {} adapter_optimizer != adam, not implemented!\nFunc: {}\nFile:{}'.format(
                rank, __name__, __file__))
        os._exit(0)    
    
    scaler = GradScaler() if args.use_amp else None  # 新增
        
    # Define adapter scheduler
    if not args.adapter_lr_warm_up:
        adapter_scheduler = None
        from utils.common_utils import adjust_lr
        
        adapter_lr_plan = {}
    else:
        from utils.common_utils import get_cosine_schedule_with_warmup
        adapter_scheduler = get_cosine_schedule_with_warmup(
            adapter_optimizer, args.adapter_warmup_steps, len(adapter_train_loader) * args.adapter_num_epochs)
    
    if args.cudnn_benchmark:
        cudnn.benchmark = True
        
    training_adapter_start_time = time.time()
    csv_logger = os.path.join(args.log_dir, f'adapter_accLogger_{args.curr_time}.csv')
    with open(csv_logger, 'w', newline='') as f:
        _pretrain_obj_info = [obj+'_loss' for obj in args.adapter_objective] + [obj+'_acc' for obj in args.adapter_objective]
        writer = csv.DictWriter(f, fieldnames=_pretrain_obj_info+['train_loss','SF_train_acc','SF_test_acc','zero_shot','epoch'])
        writer.writeheader() 
    for adapter_epoch in range(1, args.adapter_num_epochs + 1):
        
        if args.use_ddp and not args.partition_dataset:  # distributed training
            adapter_train_loader.sampler.set_epoch(adapter_epoch)
        
        if adapter_scheduler is None:
            if adapter_epoch in adapter_lr_plan:
                adjust_lr(adapter_optimizer, adapter_lr_plan[adapter_epoch])
        
        torch.cuda.empty_cache()
        
        #################################
        # --- train adapter for one epoch
        #################################
        if need_logging(args):
            logger.info('Rank: {} '.format(rank) + '='*90)
        train_adapter_for_one_epoch_start_time = time.time()
        adapter_acc, adapter_loss = train_adapter_for_one_epoch(
            args, logger, 
            adapter_train_loader, adapter_model, 
            adapter_criterion, adapter_optimizer, adapter_scheduler, 
            adapter_epoch,scaler)
        if need_logging(args):
            logger.info(f'Adapter Epoch: {adapter_epoch} Train Loss: {adapter_loss} Train Acc: {adapter_acc}')
            logger.info("Rank: {} Finished training < adapter > adapter_epoch-{}, took {} seconds".format(
                rank, adapter_epoch, round(time.time() - train_adapter_for_one_epoch_start_time, 2)))
            logger.info('Rank: {} '.format(rank) + '='*60)

        #################################
        # --- test adapter for one epoch
        #################################
        

        if need_logging(args):
            logger.info('Rank: {} '.format(rank) + '='*90)
        
        # save adapter checkpoint
        if adapter_epoch >= args.adapter_start_save_epoch and adapter_epoch % args.adapter_save_freq == 0:
            ckp_save_path = os.path.join(args.checkpoint_dir, f'Adapter_{adapter_epoch}.pth')
            if args.use_ddp:  # distributed training
                if is_main_process():
                    save_adapter_checkpoint(args, logger, adapter_epoch, adapter_model.module.adapter, adapter_optimizer,ckp_save_path)
                dist.barrier()
            else:
                save_adapter_checkpoint(args, logger, adapter_epoch, adapter_model.module.adapter, adapter_optimizer,ckp_save_path)
        

    if need_logging(args):
        logger.info('\n\n\n' + 'Rank: {} '.format(rank) + '#'*90)       
        logger.info("Rank: {} Finished training < adapter > for all epochs, took {} seconds".format(
            rank, round(time.time() - training_adapter_start_time, 2)))                    
    

    return




def save_adapter_checkpoint(args, logger, adapter_epoch, adapter_model, adapter_optimizer,filename):
    
    save_checkpoint(
        {'cfg': args, 
         'epoch': adapter_epoch,
         'state_dict': adapter_model.module.state_dict() if hasattr(
             adapter_model, 'module') else adapter_model.state_dict(),
         'optimizer': adapter_optimizer.state_dict()
        },  
        False,
        dir=args.checkpoint_dir, 
        name='Adapter-' + args.curr_time,
        filename = filename # -{adapter_epoch}
    )
    
    if need_logging(args):
        logger.info('Rank: {} Checkpoint saved in {}'.format(
            args.rank,
            os.path.abspath(filename)
        ))
    return

from torch.cuda.amp import autocast, GradScaler 

# 修改后的 train_adapter_for_one_epoch 函数
def train_adapter_for_one_epoch(
    args, logger, 
    train_loader, model, criterion, optimizer, scheduler, epoch,scaler=None):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = dict() 
    acc_meter = dict()
    

    for obj in args.adapter_objective:
        loss_meter[obj] = AverageMeter()  
        acc_meter[obj] = AverageMeter()
    
    model.train()

    for obj in args.adapter_objective:
        if type(criterion[obj]) == list:
            for criterion_item in criterion[obj]:
                criterion_item.train() 
        else:  
            criterion[obj].train() 
    
    batch_start_time = time.time()
    load_data_stime = time.time() 
    for i, batch_data in enumerate(train_loader):

        segment_video_feat = batch_data[0].to(args.device)
        pseudo_labels = batch_data[1:1+len(args.adapter_objective)]
        assert len(pseudo_labels) == len(args.adapter_objective)
        mask = None if len(batch_data) == 1+ len(args.adapter_objective) else batch_data[-1].to(args.device)
        for i in range(len(pseudo_labels)):
            pseudo_labels[i] = pseudo_labels[i].to(args.device)

        data_time.update(time.time() - batch_start_time)
        bs = len(batch_data)
        optimizer.zero_grad()

        with autocast(enabled=args.use_amp):  
            t1 = time.time() 
            output = model(segment_video_feat, mask)
            
            batch_loss_record = dict()
            for j, obj in enumerate(args.adapter_objective):
                answer = output[j]
                pseudo_label = pseudo_labels[j]

                assert pseudo_label.sum().item() != 0

                if type(criterion[obj]) == list:
                    loss = 0 
                    acc = 0
                    for k, criterion_item in enumerate(criterion[obj]):
                        pred, label = valid_mask(answer[k], pseudo_label[:, k])
                        loss += criterion_item(pred, label)
                        
                        y_true = label.long()
                        sigmoid_logits = torch.sigmoid(pred)
                        y_pred = sigmoid_logits

                        t3 = time.time() 
                        acc += multilabel_cls_exact_match(y_pred, y_true) if epoch % 10 == 0 else np.float32(0)
                    acc /= len(criterion[obj])
                else:  
                    pred, label = valid_mask(answer, pseudo_label)
                    loss = criterion[obj](answer, pseudo_label)
                    
                    if args.classify_type == 'single_label':
                        acc = accuracy(pred, label, topk=(1,))
                    elif args.classify_type == 'multi_label':
                        y_true = label.long()
                        sigmoid_logits = torch.sigmoid(pred)
                        y_pred = sigmoid_logits

                        acc = multilabel_cls_exact_match(y_pred, y_true) if epoch % 10 == 0 else np.float32(0) 

                loss_meter[obj].update(loss.item(), bs)
                acc_meter[obj].update(acc.item(), bs) 

                batch_loss_record['loss_' + obj] = loss
                batch_loss_record['acc_' + obj] = acc

            loss_ratio = [getattr(args, f'{obj}_loss_ratio') for obj in args.adapter_objective]
            loss_value = [batch_loss_record[f'loss_{obj}'] for obj in args.adapter_objective]
            loss_thisbatch = sum([a * b for a, b in zip(loss_ratio, loss_value)])
        if args.use_amp:  
            scaler.scale(loss_thisbatch).backward() 
            scaler.step(optimizer)                  
            scaler.update()                         
        else:
            loss_thisbatch.backward()
            # optimizer.step()
        if scheduler:
            scheduler.step()
        
        if args.use_wandb and is_main_process():
            grad = calc_grad(model)
        
        batch_time.update(time.time() - batch_start_time)
        batch_start_time = time.time()
        
    avg_acc = {} 
    avg_loss = {}
    if args.use_ddp and not args.partition_dataset:
        for obj in args.adapter_objective:
            avg_acc[obj] = global_meters_all_avg(args, acc_meter[obj].avg)
            avg_loss[obj] = global_meters_all_avg(args, loss_meter[obj].avg)
    else:
        avg_acc = {k: v.avg for k, v in acc_meter.items()}
        avg_loss = {k: v.avg for k, v in loss_meter.items()}
    return avg_acc, avg_loss  

from utils.common_utils import Bunch
import yaml 
if __name__ == '__main__':
    
    args = get_args_parser()
    set_seed(args.seed)
    if args.use_ddp:
        rank = get_rank()
    else:
        rank = 0
    args.rank = rank
   
        
    if args.use_ddp and args.ddp_log_each_rank: 
        args.log_dir = os.path.join(args.log_dir, str(rank))
        os.makedirs(args.log_dir, exist_ok=True)

    logfile_path = os.path.abspath(
        os.path.join(args.log_dir, f'{args.curr_time}_trainAdapter_{args.notes}.log'))
        
    if need_logging(args):
        logger = getLogger(name=__name__, path=logfile_path)
    else:
        logger = None
    
    main_train_adapter(args,logger)
     