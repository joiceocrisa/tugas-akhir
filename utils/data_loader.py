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
    
    data = df[feature_col].values.reshape(-1, 1)
    
    # Split data
    n_total = len(data)
    train_end = int(n_total * train_ratio)
    val_end = train_end + int(n_total * val_ratio)
    
    raw_train = data[:train_end]
    raw_val = data[train_end:val_end]
    raw_test = data[val_end:]
    
    # Scaling (Hanya Fit pada Training Data)
    scaler = None
    if scale:
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit hanya pada training untuk mencegah data leakage
        train_data = scaler.fit_transform(raw_train)
        # Gunakan parameter dari train untuk transform val & test
        val_data = scaler.transform(raw_val)
        test_data = scaler.transform(raw_test)
    else:
        train_data, val_data, test_data = raw_train, raw_val, raw_test

    # sliding window
    def create_sequences(dataset, s_len, p_len):
        X_seq, y_seq = [], []
        for i in range(len(dataset) - s_len - p_len + 1):
            X_seq.append(dataset[i:i + s_len])
            y_seq.append(dataset[i + s_len:i + s_len + p_len])
        return np.array(X_seq), np.array(y_seq)

    # buat sequences untuk masing-masing bagian secara terpisah
    X_train, y_train = create_sequences(train_data, seq_len, pred_len)
    X_val, y_val = create_sequences(val_data, seq_len, pred_len)
    X_test, y_test = create_sequences(test_data, seq_len, pred_len)

    # Reshape y ke (samples, pred_len) jika perlu
    if y_train.ndim == 3:
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
    if y_val.ndim == 3:
        y_val = y_val.reshape(y_val.shape[0], y_val.shape[1])
    if y_test.ndim == 3:
        y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
