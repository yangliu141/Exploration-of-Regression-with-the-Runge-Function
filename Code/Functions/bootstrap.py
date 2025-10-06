d

def bootstrap(n_bootstraps, X, y, evaluate_model):
    """
        Performs n_bootstraps bootstraps.

        Parameters:
            n_bootstraps (int): number of bootstraps
            X (np.ndarray): feature matrix
            y (np.ndarray): target vector
            evaluate_model (function): a function with signature
                evaluate_model(x_train, y_train, x_test, y_test) -> float
                that returns a performance score for the model.

        Returns:
            float: mean performance score across bootstrap samples
            np.ndarray: array of scores for each bootstrap sample
    """
    for b in range(n_bootstraps):
        X_re, y_re = resample(X, y)





