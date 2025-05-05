import torch
from omegaconf import DictConfig # type: ignore
import os
import random
import numpy as np

def getDevice(cfg: DictConfig):
    # Dynamically determine the device and store as a string
    cfg.DEVICE_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device(cfg.DEVICE_str)
    print("Using device:", DEVICE)
    return DEVICE

def check_cuda(cfg_base: DictConfig):
    print('---------- Checking Version of Virtual Environment before starting ---------------')
    print("Torch version:",torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("Is CUDA enabled?",torch.cuda.is_available())

    DEVICE = torch.device(cfg_base.DEVICE_str)
    print("Using device:", DEVICE)

    #Additional Info when using cuda
    if DEVICE.type == 'cuda':
        print(f'current_device Name : {torch.cuda.get_device_name(0)}')
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    else:
        print("CUDA is not available. Running on CPU.")

def seed_everything(cfg_base: DictConfig):
    seed = cfg_base.SEED
    print(f'seed from hydra config file is {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

