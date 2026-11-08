from utils.common_utils import need_logging


def create_model(args, logger, model_name, adapter_prediction=True):
    if  model_name == 'adapter_mlp':
        if hasattr(args, "adapter_inference") and args.adapter_inference:
            from models.Adapter.adapter_inference import Adapter
        else:
            from models.Adapter.adapter import Adapter
        model = Adapter(args, logger, adapter_prediction)
    elif model_name == 'adapter_progress':
        from models.Adapter.adapter_progress import Adapter
        model = Adapter(args, logger, adapter_prediction)
    elif model_name == 'adapter_tx':
        from models.Adapter.adapter_tx import Adapter
        model = Adapter(args, logger, adapter_prediction)
    elif 'downstream' in model_name:
        if args.downstream_task_name == 'task_cls':
            from models.Downstream.task_head_task_cls import Task_Head
        elif args.downstream_task_name == 'step_cls':
            from models.Downstream.task_head_step_cls import Task_Head
        elif args.downstream_task_name == 'step_forecasting':
            from models.Downstream.task_head_step_forecasting import Task_Head
        model = Task_Head(args, logger)
    elif model_name == 'test':
        from models.test_models import test_model
        model = test_model(args, logger)

    else:
        raise ValueError("Model {} not recognized.".format(args.adapter_name))


    if need_logging(args):
        logger.info(model)
        logger.info("--> model {} was created".format(model_name))

    return model
