import os
import sys
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import hydra
from omegaconf import DictConfig # type: ignore
# import numpy as np

from dataset.preprocess import *
from helper.misc_helper import *

def process_dataset(cfg: DictConfig):
    print(f'{cfg.run_type} {cfg.dataset} with {cfg.model}')
    preprocess_dataset(cfg)


@hydra.main(version_base=None, config_path="../conf", config_name="config")

def main(cfg: DictConfig):
    print(f'**** Main file ****')
    seed_everything(cfg)
    DEVICE = getDevice(cfg)
    check_cuda(cfg)
    process_dataset(cfg)
    print(f'**** End ****')

if __name__ == "__main__":
    main()