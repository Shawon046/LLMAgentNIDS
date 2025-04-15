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
from model.accuracy_metrics import *


def load_processed_dataset(cfg: DictConfig) -> None:


    print(f"{cfg.run_type.capitalize()} {cfg.model} model on {cfg.dataset}")
    
    print(f"Starting point: {cfg.dataset_dir} ")
    train_dataloader = get_dataloader(cfg, True) 
    test_dataloader = get_dataloader(cfg, False) 
    return train_dataloader, test_dataloader

def baseline_model_train_test(cfg, train_dataloader, test_dataloader):
    # Use this dataloaders to train and test models
    result_dict_test = baseline_traditional_ml_models(cfg, train_dataloader, test_dataloader)
    compare_accuracy_metrics(cfg, result_dict_test)

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



#TODO: Tauhid Vai and Sabiha Apu
def llm_based_agent_train_test(cfg, train_dataloader,test_dataloader):
    print('Implement LLM based agent here')

@hydra.main(version_base=None, config_path="../conf", config_name="config")

def main(cfg: DictConfig):
    print(f'**** Main file ****')
    seed_everything(cfg)
    DEVICE = getDevice(cfg)
    check_cuda(cfg)
    train_dataloader, test_dataloader = load_processed_dataset(cfg)
    baseline_model_train_test(cfg, train_dataloader, test_dataloader)
    
    #TODO: Tauhid Vai and Sabiha Apu
    llm_based_agent_train_test(cfg, train_dataloader,test_dataloader)
    print(f'**** End ****')

if __name__ == "__main__":
    main()