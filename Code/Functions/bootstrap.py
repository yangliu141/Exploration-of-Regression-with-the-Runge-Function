from sklearn.utils import resample
import numpy as np
from Functions.gradientdescent_lib import MSE
from typing import Callable

def bootstrap(n_bootstraps : int, X_train : np.array, y_train : np.array, X_test : np.array, y_test : np.array, evaluate_model : Callable) -> tuple[float, float, float]:
    """
        Performs n_bootstraps bootstraps.

        Parameters:
            n_bootstraps (int): number of bootstraps
            X (np.ndarray): feature matrix
            y (np.ndarray): target vector
            evaluate_model (function): a function with signature
                evaluate_model(x_train, y_train, x_test, y_test) -> float, array
                that returns a performance score for the model and the array of the predictions

        Returns:
            float: mean performance score across bootstrap samples
            np.ndarray: array of scores for each bootstrap sample
    """

    y_pred = np.empty((n_bootstraps, len(y_test)))

    for b in range(n_bootstraps):
        X_train_re, y_train_re = resample(X_train, y_train)
        score, predictions = evaluate_model(X_train_re, y_train_re, X_test, y_test)    
        y_pred[b, :] = predictions


    predictions_mean = np.mean(y_pred, axis=0)

    mse = np.mean(np.mean((y_pred - y_test)**2, axis = 1))
    bias = np.mean((y_test - predictions_mean)**2)
    variance = np.mean(np.mean((y_pred - predictions_mean)**2, axis = 0))
        
    return mse, bias, variance




