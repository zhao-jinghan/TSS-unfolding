CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
--nnodes=1 --nproc_per_node=8  --master_port=29500  engines/main_CoP_pretrain.py  \
--cfg configs/CoP_pretrain/CoP_pretrain_by_stage.yml \
--use_ddp 1 \
--use_wandb 0 \
--notes N \
--target_stage task \
--load_pretrained 0