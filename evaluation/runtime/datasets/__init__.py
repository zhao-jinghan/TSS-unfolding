import os
import pickle
import copy

def return_dataset(args, logger, dataset_name, dataset_split='train'):
    # from datasets.test_dataset import Test_dataset
    # return Test_dataset(args, logger)
    if 'HowTo100M' in dataset_name:
        # if args.target_stage is not None and  '_' in args.target_stage:
        #     from datasets.ht100m_mix_stage import HT100M
        # else:
        #     from datasets.ht100m import HT100M
        from datasets.ht100m import HT100M
        return HT100M(args, logger)
    if dataset_name == 'CoP_Ex':
        from datasets.ht100m import HT100M
        CoP_ex_config = copy.deepcopy(args)
        CoP_ex_config.dataset_dir['CoP'] = CoP_ex_config.dataset_dir['CoP_Ex']

        return HT100M(CoP_ex_config, logger,Ex=True)

    elif dataset_name == 'CrossTask':
        if 'task' in args.downstream_task_name:
            if (not os.path.exists(
                os.path.join(args.cross_task_s3d_feat_dir, '{}_train_split.pickle'.format(
                    args.downstream_task_name)))) or (
                not os.path.exists(os.path.join(args.cross_task_s3d_feat_dir, '{}_test_split.pickle'.format(
                    args.downstream_task_name)))):

                from datasets.cross_task import get_task_cls_train_test_splits
                train_split, test_split = get_task_cls_train_test_splits(
                    feat_dir=args.cross_task_s3d_feat_dir,
                    annotation_dir=args.cross_task_annot_dir, train_ratio=0.8)
                with open(os.path.join(args.cross_task_s3d_feat_dir, '{}_train_split.pickle'.format(
                    args.downstream_task_name)), 'wb') as f:
                    pickle.dump(train_split, f)
                with open(os.path.join(args.cross_task_s3d_feat_dir, '{}_test_split.pickle'.format(
                    args.downstream_task_name)), 'wb') as f:
                    pickle.dump(test_split, f)

        elif 'step' in args.downstream_task_name:
            if (not os.path.exists(
                os.path.join(args.cross_task_s3d_feat_dir, '{}_train_split.pickle'.format(
                    args.downstream_task_name)))) or (
                not os.path.exists(os.path.join(args.cross_task_s3d_feat_dir, '{}_test_split.pickle'.format(
                    args.downstream_task_name)))):

                from datasets.cross_task import get_step_cls_train_test_splits
                task_meta, steps, train_split, test_split = get_step_cls_train_test_splits(
                    args.cross_task_annot_dir, args.cross_task_s3d_feat_dir, train_ratio=0.8)
                with open(os.path.join(args.cross_task_s3d_feat_dir, '{}_train_split.pickle'.format(
                    args.downstream_task_name)), 'wb') as f:
                    pickle.dump(train_split, f)
                with open(os.path.join(args.cross_task_s3d_feat_dir, '{}_test_split.pickle'.format(
                    args.downstream_task_name)), 'wb') as f:
                    pickle.dump(test_split, f)
                with open(os.path.join(args.cross_task_s3d_feat_dir, '{}_task2step_meta.pickle'.format(
                    args.downstream_task_name)), 'wb') as f:
                    pickle.dump(task_meta, f)
                with open(os.path.join(args.cross_task_s3d_feat_dir, '{}_set_of_steps.pickle'.format(
                    args.downstream_task_name)), 'wb') as f:
                    pickle.dump(steps, f)

        if args.downstream_task_name == 'task_cls':
            from datasets.cross_task import CrossTask_Task_CLS as CrossTask

        elif args.downstream_task_name == 'step_cls':
            from datasets.cross_task import CrossTask_Step_CLS as CrossTask

        elif args.downstream_task_name == 'step_forecasting':
            from datasets.cross_task import CrossTask_Step_Forecast as CrossTask

        return CrossTask(args, logger, split=dataset_split)

    elif dataset_name == 'COIN':
        if args.downstream_task_name == 'task_cls':
            from datasets.coin import COIN_Task_CLS as COIN
            # from datasets.test_dataset import Test_dataset
            # return Test_dataset(args, logger)
        elif args.downstream_task_name == 'step_cls':
            from datasets.coin import COIN_Step_CLS as COIN
        elif args.downstream_task_name == 'step_forecasting':
            from datasets.coin import COIN_Step_Forecast as COIN

        return COIN(args, logger, split=dataset_split)
    elif dataset_name == 'test':
        from datasets.test_dataset import Test_dataset
        return Test_dataset(args, logger)
