import os
import sys
# # Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from pathlib import Path

import hydra
from omegaconf import DictConfig # type: ignore

from dataset.preprocess import *
from helper.misc_helper import *
from dataset.load_data import get_dataloader

def process_dataset(cfg: DictConfig):
    print(f'{cfg.run_type} {cfg.dataset} with {cfg.model}')
    preprocess_dataset(cfg)



def load_and_train_model(cfg: DictConfig) -> None:

    # # Update config with source directory
    # source_dir = Path(__file__).resolve().parent
    # cfg = update_abs_path(cfg_base.copy(), source_dir)
    print(f"{cfg.run_type.capitalize()} {cfg.model} model on {cfg.dataset}")

    
    print(f"Starting point: {cfg.dataset_dir} ")
    dataloader = get_dataloader(cfg) 

@hydra.main(version_base=None, config_path="../conf", config_name="config")

def main(cfg: DictConfig):
    print(f'**** Main file ****')
    seed_everything(cfg)
    DEVICE = getDevice(cfg)
    check_cuda(cfg)
    process_dataset(cfg)
    load_and_train_model(cfg)
    print(f'**** End ****')

if __name__ == "__main__":
    main()