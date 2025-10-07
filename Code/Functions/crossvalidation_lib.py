from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from typing import Callable, Tuple

def cross_validation(k: int, x: np.ndarray, y: np.ndarray, evaluate_model: Callable[..., Tuple[float, np.ndarray]]) -> Tuple[float, np.ndarray, float, np.ndarray]:
    """
        Perform k-fold cross-validation.

        Parameters:
            k (int): number of folds
            x (np.ndarray): feature matrix
            y (np.ndarray): target vector
            evaluate_model (function): a function with signature
                evaluate_model(x_train, y_train, x_test, y_test) -> float, array
                that returns a performance score for the model, and the predictions made for the test data

        Returns:
            float: mean performance score across folds
            np.ndarray: array of scores for each fold
    """
    scores_kfold = np.empty(k)
    scores_R2 = np.empty(k)

    kfold = KFold(n_splits = k, shuffle = True, random_state = 10)
    
    for fold, (train_idx, test_idx) in enumerate(kfold.split(x)):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        mse, r2, predictions = evaluate_model(x_train, y_train, x_test, y_test)
        scores_kfold[fold] = mse
        scores_R2[fold] = r2
    
    return np.mean(scores_kfold), scores_kfold, np.mean(scores_R2), scores_R2

def convertToTable(results : np.array, file_path : str, columns : list, description : str, name : str, write_type : str = "w") -> None:
    """
    Converts the input to a Latex table and saves it to a file. 
    """
    df = pd.DataFrame(results, index=["OLS", "Ridge", "Lasso"], columns=columns)

    with open("figures\\"+file_path, write_type) as f:
        f.write("\\newcommand{\\crossValidatoin}{\n")
        f.write(f"% ---- Cross validation ----\n")
        f.write(f"\\begin{{table}}[H]\n\\centering\n\\caption"+"{"+f" {description}\n"+"}"+"\\label{tab:"+name+"}")
        f.write(df.to_latex(float_format="%.4f"))
        f.write("\\end{table}\n")
        f.write("}")