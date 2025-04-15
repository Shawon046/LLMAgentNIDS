import torch
# from .datasets import *
# from .datasets.timeseries.get_timeseries import get_timeseries_datasets
# from .datasets.vision.get_vision import get_vision_datasets
# from .get_tabular import *
# from robids.helper.visualize_helper import save_and_display_image_grid

import matplotlib.pyplot as plt
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

# from .load_nids_data import get_nids_data
# from ..helper.dataset_helper import *
from helper.dataset_helper import *
from .load_nsl_data import *


def get_dataset(cfg, train_split):
    """
    Retrieve datasets based on the modality specified in the configuration.
    
    Args:
        cfg (DictConfig): Configuration object that contains dataset settings.

    Returns:
        tuple: dataset_train, dataset_test, dataset_valid
    Raises:
        ValueError: If the dataset modality is unsupported.
    """

    # Get the dataset name from the config
    dataset_type = cfg.dataset_type
    # Check if the dataset name is in the mapping
    if dataset_type == 'nids':
        # X, y = get_nids_data(cfg)

        # train_split = True if cfg.run_type == 'train' else False
        dataset_name = cfg.dataset
        if dataset_name == 'nsl-kdd':
            X, y = get_nsl_dataset(cfg, train_split)

        else:
            print("Data set not supported yet!")
            raise ValueError(f"Unsupported dataset name: {dataset_name}")

        # Check the shapes of X and y
        print(f"Shape of X (images): {X.shape}")
        print(f"Shape of y (labels): {y.shape}")

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
    print("*"*25+ f"  Loaded Tabular data {type(X)} "+"*"*25)


    dataset = CreateDataset(X, y) 
    print("*"*25+ f"  Loaded Dataset {type(dataset)} "+"*"*25)
    return dataset




def get_dataloader(cfg, train_split):
    """
    Unified function to get the appropriate data loader based on the dataset type.
    
    Args:
        cfg (dict): Configuration dictionary that contains dataset type and paths.
        
    Returns:
        train_loader, valid_loader, test_loader: DataLoader objects for training, validation, and testing.
    """
    print(f'**Inside Load data file**')
    batch_size = cfg.batch_size
    shuffle = True if cfg.run_type == 'train' else False
    dataset = get_dataset(cfg, train_split)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=batch_size,
                                                shuffle=shuffle)
    images_batch, _ = next(iter(dataloader))  # Get a batch of images
    
    # save_and_display_image_grid(cfg, images_batch)
    print_label_counts(dataloader) 
    print(dataloader)
    print("*"*25+ f"  Loaded Dataloader: {type(dataloader)} "+"*"*25)
    return dataloader

