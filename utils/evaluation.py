import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Optional
import os


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate evaluation metrics for predictions.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary containing metrics: MAE, MSE, RMSE, MAPE, R2
    """
    # Flatten arrays if needed
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Remove any NaN or Inf values
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # # R2 Score
    # r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
    }
    
    return metrics


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Stock Price Prediction",
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot predictions against true values.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Flatten if needed
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Create time index
    time_steps = range(len(y_true))
    
    plt.plot(time_steps, y_true, label='True Values', linewidth=2, alpha=0.7)
    plt.plot(time_steps, y_pred, label='Predictions', linewidth=2, alpha=0.7)
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Stock Price', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_comparison(
    y_true: np.ndarray,
    y_pred_lstm: np.ndarray,
    y_pred_fedformer: np.ndarray,
    title: str = "Model Comparison",
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    Plot comparison between LSTM and FEDFormer predictions.
    
    Args:
        y_true: True values
        y_pred_lstm: LSTM predictions
        y_pred_fedformer: FEDFormer predictions
        title: Plot title
        save_path: Path to save the plot (optional)
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=(14, 8))
    
    # Flatten if needed
    y_true = y_true.flatten()
    y_pred_lstm = y_pred_lstm.flatten()
    y_pred_fedformer = y_pred_fedformer.flatten()
    
    # Create time index
    time_steps = range(len(y_true))
    
    plt.plot(time_steps, y_true, label='True Values', linewidth=2.5, alpha=0.8, color='black')
    plt.plot(time_steps, y_pred_lstm, label='LSTM Predictions', linewidth=2, alpha=0.7, linestyle='--')
    plt.plot(time_steps, y_pred_fedformer, label='FEDFormer Predictions', linewidth=2, alpha=0.7, linestyle='-.')
    
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Stock Price', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def print_metrics(metrics: dict, model_name: str = "Model") -> None:
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        model_name: Name of the model
    """
    print(f"\n{model_name} Metrics:")
    print("-" * 40)
    for metric_name, value in metrics.items():
        if metric_name == 'MAPE':
            print(f"{metric_name:10s}: {value:.4f}%")
        elif metric_name == 'R2':
            print(f"{metric_name:10s}: {value:.4f}")
        else:
            print(f"{metric_name:10s}: {value:.4f}")
    print("-" * 40)

