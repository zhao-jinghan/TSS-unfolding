from utils.common_utils import need_logging


def create_model(args, logger, model_name, adapter_prediction=True):
    if  model_name == 'adapter_mlp':
        
        from models.Adapter.adapter import Adapter
        model = Adapter(args, logger, adapter_prediction)
    
    if need_logging(args):
        logger.info(model)
        logger.info("--> model {} was created".format(model_name))

    return model

