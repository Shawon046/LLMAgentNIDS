# TODO: Shahedul

from torchvision import datasets, transforms


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from .preprocess_nsl_data import *

class NSLKDDDataset(Dataset):
    def __init__(self, cfg, train=True, transform=None):
        """
        Custom Dataset for NSL-KDD
        
        Args:
        - data_path: Path to the NSL-KDD dataset CSV files
        - train: Whether to load training or testing data
        - transform: Optional transform to be applied on a sample
        """
        self.cfg = cfg
        # Determine which files to load based on train/test split
        if train:
            train_file = f"{cfg.dataset_dir}/KDDTrain+.csv"
        else:
            train_file = f"{cfg.dataset_dir}/KDDTest+.csv"

        # col_names = ["duration","protocol_type","service","flag","src_bytes",
        #         "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
        #         "logged_in","num_compromised","root_shell","su_attempted","num_root",
        #         "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
        #         "is_host_login","is_guest_login","count","srv_count","serror_rate",
        #         "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
        #         "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        #         "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
        #         "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
        #         "dst_host_rerror_rate","dst_host_srv_rerror_rate","attack_type", "difficulty_score"]
        # 
        col_names = cfg.columns
        # print(col_names)
        print(len(col_names))
        # Read the dataset
        df = pd.read_csv(train_file, sep=",", header=None, names=col_names)

        print(df.head())

        # Drop Unnecessary Column
        drop_column(cfg, df)

        # self.get_label_distribution(train, df)
        self.get_categorical_distribution(train, df)
        # cat_dict = cfg.dataset.cat_columns
        # print(f'Cat columns from config file {cat_dict}')

        # # Separate features and labels
        # X, y = self.separate_X_y_from_df(df)
        # print(f'Shape X {X.shape} and y {y.shape}')


        # Preprocess the data
        if train is True:
            X_scaled, y_encoded = preprocess_train_data(cfg, df)
        else:
            X_scaled, y_encoded =preprocess_test_data(cfg, df) 

        # df = self.preprocess_data(df)
        
        # Split features and labels
        self.X = X_scaled
        self.y = y_encoded

        self.transform = transform

    def get_all_data(self):
        return (self.X, self.y)

    
    def get_label_distribution(self, train, df):
        # Label Distribution of Training and Test set
        if train:
            print('Label distribution Training set:')
        else: 
            print('Label distribution Testing set:')
        print(df['label'].value_counts())

    def get_categorical_distribution(self, train, df):
        # Label Distribution of Training and Test set
        if train:
            print('Categorical distribution in Training set:')
        else: 
            print('Categorical distribution in Testing set:')
        for col_name in df.columns:
            if df[col_name].dtypes == 'object' :
                unique_cat = len(df[col_name].unique())
                print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

        # see how distributed the feature service is, it is evenly distributed and therefore we need to make dummies for all.
        print('\nDistribution of categories in service:')
        print(df['service'].value_counts().sort_values(ascending=False).head())


    

    def preprocess_data(self, df):
        """
        Preprocess the NSL-KDD dataset
        - Perform one-hot encoding for categorical features
        - Normalize numerical features
        """
        # Identify categorical and numerical columns
        # This is a placeholder and might need adjustment based on the exact NSL-KDD dataset structure
        categorical_cols = []
        col_count = 0
        for col_name in df.columns:
            col_count +=1 
            if df[col_name].dtypes == 'object' :
                # unique_cat = len(df[col_name].unique())
                categorical_cols.append(col_count)

        numerical_cols = list(set(range(df.shape[1])) - set(categorical_cols + [-1]))
        
        # One-hot encode categorical columns
        df_encoded = pd.get_dummies(df, columns=categorical_cols)
        
        # Normalize numerical columns
        for col in numerical_cols:
            df_encoded[col] = (df_encoded[col] - df_encoded[col].mean()) / df_encoded[col].std()
        
        return df_encoded

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = torch.tensor(self.X[idx], dtype=torch.float32)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        
        # if self.transform:
        #     features = self.transform(features)
        
        return features, label
    


def get_nsl_dataset(cfg, train_split):
    """
    Load NSL-KDD dataset
    
    Args:
    - cfg: Configuration object with data_dir attribute
    - train_split: Boolean to determine train or test dataset
    
    Returns:
    - NSL-KDD Dataset
    """
    print(f'**Inside Load NSL data file**')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = NSLKDDDataset(
        cfg=cfg, 
        train=train_split, 
        transform=transform
    )

    # Get all data at once
    X_tensor, y_tensor = dataset.get_all_data()
    return X_tensor, y_tensor