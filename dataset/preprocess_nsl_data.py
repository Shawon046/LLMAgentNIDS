from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


def binarize_labels(labels):
    """
    Convert multi-class labels to binary (0 for normal, 1 for attack).
    
    Parameters:
    -----------
    labels : array-like
        Original labels from the dataset
    
    Returns:
    --------
    numpy.ndarray: Binary labels (0 for normal, 1 for attack)
    """
    return np.where(labels == 'normal', 0, 1)

def create_attack_encoders(y_train, scaler_file_dir):
    # Not necessary for binary attack levels
    # Encode attack types
    attack_encoder = LabelEncoder()
    y_train_encoded = attack_encoder.fit_transform(y_train)
    # Save attack type encoder
    joblib.dump(attack_encoder, os.path.join(scaler_file_dir, 'attack_encoder.joblib'))
    return y_train_encoded

def load_attack_encoders(y_test, scaler_file_dir):
    # Load attack type encoder
    attack_encoder_path = os.path.join(scaler_file_dir, 'attack_encoder.joblib')
    if not os.path.exists(attack_encoder_path):
        raise FileNotFoundError("Attack encoder not found. Run preprocess_train_data first.")
    attack_encoder = joblib.load(attack_encoder_path)
    # Encode attack types using pre-fitted encoder
    y_test_encoded = attack_encoder.transform(y_test)
    return y_test_encoded

def create_label_encoders(cfg, train_data, scaler_file_dir):
    """
    Create and save label encoders for categorical columns.
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Training dataset
    output_dir : str, optional
        Directory to save label encoder files
    
    Returns:
    --------
    dict: Label encoders for each categorical column
    """
    print(f'Label Encoder is going to be saved to {scaler_file_dir}')
    os.makedirs(scaler_file_dir, exist_ok=True)
    
    # Categorical columns to encode
    cat_columns = cfg.categorical_columns
    label_encoders = {}
    
    for col in cat_columns:
        # Create and fit LabelEncoder
        le = LabelEncoder()
        le.fit(train_data[col])
        
        # Save the encoder
        joblib.dump(le, os.path.join(scaler_file_dir, f'{col}_encoder.joblib'))
        
        # Store in dictionary for potential further use
        label_encoders[col] = le
    
    return label_encoders

def load_label_encoders(cfg, scaler_file_dir):
    """
    Load previously saved label encoders.
    
    Parameters:
    -----------
    output_dir : str, optional
        Directory where label encoder files are stored
    
    Returns:
    --------
    dict: Loaded label encoders for each categorical column
    """
    scaler_file_dir = f'{cfg.scaler_dir}/label_encoders'
    print(f'Label Encoder is loaded from {scaler_file_dir}')
    cat_columns = cfg.categorical_columns
    label_encoders = {}
    
    for col in cat_columns:
        encoder_path = os.path.join(scaler_file_dir, f'{col}_encoder.joblib')
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(f"Encoder for {col} not found. Run create_label_encoders first.")
        
        # Load the encoder
        label_encoders[col] = joblib.load(encoder_path)
    
    return label_encoders


def preprocess_train_data(cfg, train_data):
    """
    Preprocess training data and create label encoders.
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Training dataset
    output_dir : str, optional
        Directory to save label encoder files
    
    Returns:
    --------
    tuple: Preprocessed training data and scaler
    """
    print(f'Preprocess training data and scaler')
    scaler_file_dir = f'{cfg.scaler_dir}/label_encoders'
    # Create label encoders and save them
    create_label_encoders(cfg, train_data, scaler_file_dir)
    
    # Separate features and labels
    X_train, y_train = separate_X_y_from_df(cfg, train_data)

    # Binarize the labels
    y_train_binary = binarize_labels(y_train)

    # print(f'Len of y train : {len(y_train_binary)}')
    # Load pre-saved label encoders
    label_encoders = load_label_encoders(cfg, scaler_file_dir)
    
    # Categorical column encoding using pre-saved encoders
    cat_columns = cfg.categorical_columns
    for col in cat_columns:
        X_train[col] = label_encoders[col].transform(X_train[col])
    
    # Binary columns encoding
    binary_columns = cfg.binary_columns
    for col in binary_columns:
        X_train[col] = X_train[col].astype(int)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Save the scaler
    joblib.dump(scaler, os.path.join(scaler_file_dir, 'standard_scaler.joblib'))
    
    # Not necessary for binary attack levels
    # create_attack_encoders(y_train, scaler_file_dir)

    print(f'Training: Shape X {X_train_scaled.shape} and y {y_train_binary.shape}')
    return X_train_scaled, y_train_binary

def preprocess_test_data(cfg, test_data):
    """
    Preprocess test data using pre-saved encoders.
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Test dataset
    output_dir : str, optional
        Directory where label encoder files are stored
    
    Returns:
    --------
    tuple: Preprocessed test data
    """
    print(f'Preprocess testing data and scaler')
    scaler_file_dir = f'{cfg.scaler_dir}/label_encoders'
    print(f'Label Encoder is loaded from {scaler_file_dir}')
    # Separate features and labels
    X_test = test_data.drop('attack_type', axis=1)
    y_test = test_data['attack_type']
    
    # Binarize the labels
    y_test_binary = binarize_labels(y_test)

    # Load pre-saved label encoders
    label_encoders = load_label_encoders(cfg, scaler_file_dir)
    
    # Load pre-saved scaler
    scaler_path = os.path.join(scaler_file_dir, 'standard_scaler.joblib')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found. Run preprocess_train_data first.")
    else: 
        print(f'Scaler found at {scaler_file_dir}/standard_scaler.joblib')
    scaler = joblib.load(scaler_path)
    
    # Categorical column encoding using pre-saved encoders
    cat_columns = cfg.categorical_columns
    for col in cat_columns:
        X_test[col] = label_encoders[col].transform(X_test[col])
    
    # Binary columns encoding
    binary_columns = cfg.binary_columns
    for col in binary_columns:
        X_test[col] = X_test[col].astype(int)
    
    # Feature scaling using pre-fitted scaler
    X_test_scaled = scaler.transform(X_test)
    
    # Not necessary for binary attack levels
    # load_attack_encoders(y_test, scaler_file_dir)
    print(f'Testing: Shape X {X_test_scaled.shape} and y {y_test_binary.shape}')
    return X_test_scaled, y_test_binary


def separate_X_y_from_df(cfg, df):
    """
    Separate Separate features and labels
    Args:
    - df: Dataframe created from the NSL-KDD dataset CSV files
    """
    X = df.drop(cfg.target_col, axis=1)
    y = df[cfg.target_col]
    print(f'Shape X {X.shape} and y {y.shape}')
    return X, y

def drop_column(cfg, df, col_name = None):
        # We have found out that feature difficulty_score is not needed for the model to learn intrusion. So we can drop this column
        if col_name is None:
            col_name = cfg.redundant_col
        df = df.drop(col_name, axis=1)
        print(df.head())


def scale_dataset(data_type, features, labels, df_time, scaler_dir):
    print(data_type)

    feat_code = ("").join([f[0] for f in features])
    scaler_name = f"scaler_{feat_code}.save" 
    scaler_file_dir = scaler_dir / scaler_name

    if data_type == 'training':
        scaler = OneHotEncoder()
        scaler = scaler.fit(df_time[features].values)
        joblib.dump(scaler, scaler_file_dir) 
        print("Scaler saved!!")

    if data_type == 'testing':
        scaler = joblib.load(scaler_file_dir) 

    df_time_scaled_feat = pd.DataFrame(scaler.transform(df_time[features].values), columns=features)
    df_time_scaled = pd.concat([df_time_scaled_feat, df_time[labels]], axis= 1)
    return df_time_scaled
