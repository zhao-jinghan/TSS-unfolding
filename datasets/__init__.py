import os
import pickle
import copy

def return_dataset(args, logger, dataset_name, dataset_split='train'):
    from datasets.ht100m import HT100M    
    return HT100M(args, logger)
    