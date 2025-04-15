import os
import sys
# # Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from pathlib import Path

import hydra
from omegaconf import DictConfig # type: ignore

from helper.misc_helper import *
from dataset.load_data import get_dataloader
from model.traditional_models import *

def load_and_train_model(cfg: DictConfig) -> None:


    print(f"{cfg.run_type.capitalize()} {cfg.model} model on {cfg.dataset}")
    
    print(f"Starting point: {cfg.dataset_dir} ")
    train_dataloader = get_dataloader(cfg, True) 
    test_dataloader = get_dataloader(cfg, False) 
    # baseline_traditional_ml_models(cfg, train_dataloader, test_dataloader)

    # model, pre_trained, rem_epochs = get_model(cfg)
    # if cfg.run_type == 'train' and pre_trained:
    #     print("Model is already trained.")
    #     # return
    # elif cfg.run_type == 'test' and not pre_trained:
    #     print("Model is not trained yet.")
    #     return
       
    # print("Starting point: cfg.data_dir :", cfg.data_dir)
    # dataloader = get_dataloader(cfg) 

    # # Get training data and train model
    # if cfg_base.run_type == 'train':
    #     model = train_model(cfg, model, dataloader, rem_epochs, pre_trained)

    # elif cfg_base.run_type == 'test':
    #     results = test_model(cfg, model, dataloader)
    #     print("results :", results)


@hydra.main(version_base=None, config_path="../conf", config_name="config")

def main(cfg: DictConfig):
    print(f'**** Main file ****')
    seed_everything(cfg)
    DEVICE = getDevice(cfg)
    check_cuda(cfg)
    load_and_train_model(cfg)
    print(f'**** End ****')

if __name__ == "__main__":
    main()