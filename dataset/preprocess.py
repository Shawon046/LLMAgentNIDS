import numpy as np
# import matplotlib.pyplot as plt
from pathlib import Path
import torch

def get_folders(folder_path):
    folders = [item.name for item in Path(folder_path).iterdir() if item.is_dir()]
    return len(folders), folders

def preprocess_dataset(cfg):
    
    print(f'Preprocessing dataset at {cfg.dataset_dir}')