import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_predictions(y_true, y_pred):
    """
    Evaluate the performance of predictions.
    
    Parameters:
    y_true : list or ndarray
        Actual target values.
    y_pred : list or ndarray
        Predicted target values.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    performance_metrics = {
        'Mean Absolute Error': mae,
        'Mean Squared Error': mse,
        'Root Mean Squared Error': rmse
    }
    return performance_metrics

def plot_metrics(metrics):
    """
    Plot performance metrics.
    
    Parameters:
    metrics : dict
        Dictionary containing performance metrics.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.title('Model Performance Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.xticks(rotation=15)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('evaluation_metrics.png')
    plt.show()