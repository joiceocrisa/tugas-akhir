import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List, Optional
import os


def load_stock_data(bank_code: str, dataset_path: str = None) -> pd.DataFrame:
    """
    Load stock data from CSV file.
    
    Args:
        bank_code: Bank code (e.g., 'BBRI', 'BBNI', 'BMRI', 'BBTN')
        dataset_path: Path to the dataset folder (default: auto-detect from project root)
        
    Returns:
        DataFrame with stock data
    """
    # Auto-detect dataset path
    if dataset_path is None:
        # Get the directory where this script is located (utils/)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to project root
        project_root = os.path.dirname(script_dir)
        # Construct path to dataset folder
        dataset_path = os.path.join(project_root, 'indonesia-bank-stock-dataset')
    
    csv_path = os.path.join(dataset_path, "datasets", bank_code, f"{bank_code}.csv")
    print(csv_path)
    
    # Convert to absolute path for better error messages
    csv_path = os.path.abspath(csv_path)
    
    if not os.path.exists(csv_path):
        print(csv_path)
        raise FileNotFoundError(f"Data file not found: {csv_path}\nCurrent working directory: {os.getcwd()}\nPlease ensure the dataset folder exists.")
    
    # Read CSV with proper column names
    columns = ['Close', 'Date', 'High', 'Kode', 'Low', 'Nama', 'Open', 'Volume']
    df = pd.read_csv(csv_path, header=None, names=columns)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df

def prepare_univariate_data(
    df: pd.DataFrame,
    feature_col: str = 'Close',
    seq_len: int = 60,
    pred_len: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[MinMaxScaler]]:
    """
    Prepare univariate time series data for training.
    
    Args:
        df: DataFrame with stock data
        feature_col: Column name to use for prediction (default: 'Close')
        seq_len: Sequence length (number of time steps to look back)
        pred_len: Prediction length (number of time steps to predict)
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        scale: Whether to scale the data using MinMaxScaler
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
    """
    # Extract the feature column
    data = df[feature_col].values.reshape(-1, 1)
    
    # Scale the data
    scaler = None
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape y to (samples, pred_len)
    if y.ndim == 3:
        y = y.reshape(y.shape[0], y.shape[1])
    
    # Split data
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    X_train = X[:train_end]
    y_train = y[:train_end]
    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


