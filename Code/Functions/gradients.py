import numpy as np

class Gradients:
    class OLS:
        """
        For computing the gradient for OLS
        """
        def __call__(self, theta : np.array, X : np.array, y : np.array) -> np.array:
            return 2/X.shape[0] * (X.T @ X @ theta - X.T @ y)
        
        def __str__(self) -> str: return "OLS"
        
    class Ridge:
        """
        For computing the gradient for Ridge
        """
        def __init__(self, l : float) -> None:
            self.l = l

        def __call__(self, theta : np.array, X : np.array, y : np.array) -> np.array:
            return 2/X.shape[0] * (X.T @ X @ theta - X.T @ y) + 2*self.l*theta
        
        def __str__(self) -> str: return "Ridge"
    
    class Lasso:
        """
        For computing the gradient for Lasso
        """
        def __init__(self, l : float) -> None:
            self.l = l

        def __call__(self, theta : np.array, X : np.array, y : np.array) -> np.array:
            return 2/X.shape[0] * (X.T @ X @ theta - X.T @ y) + self.l*np.sign(theta)
        
        def __str__(self) -> str: return "Lasso"