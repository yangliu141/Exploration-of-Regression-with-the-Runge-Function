from sklearn.model_selection import KFold
import numpy as np

def cross_validation(k: int, x, y, evaluate_model):
    """
        Perform k-fold cross-validation.

        Parameters:
            k (int): number of folds
            x (np.ndarray): feature matrix
            y (np.ndarray): target vector
            evaluate_model (function): a function with signature
                evaluate_model(x_train, y_train, x_test, y_test) -> float
                that returns a performance score for the model.

        Returns:
            float: mean performance score across folds
            np.ndarray: array of scores for each fold
    """
    scores_kfold = np.empty(k)

    kfold = KFold(n_splits = k, shuffle = True, random_state = 10)
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(x)):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        performance = evaluate_model(x_train, y_train, x_test, y_test)
        scores_kfold[fold] = performance
    
    return np.mean(scores_kfold), scores_kfold