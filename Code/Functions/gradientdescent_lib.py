import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import normalize

from Functions.optimizers import Optimizers
from Functions.gradients import Gradients


def generateData(nData : int, noise : float) -> tuple[np.array, np.array, np.array, np.array]:
    """
    Generates normalized train and test data with noise for the Runge function
    """
    x = np.linspace(-1, 1, nData)
    y = 1 / (1 + 25*np.power(x, 2)) + np.random.normal(0, noise, size=nData)

    return train_test_split(normalize(x.reshape(-1, 1), axis=0, norm='max'), y, test_size = 0.2)

def featureMat(x : np.array, p : int, noIntercept : bool = True) -> np.array:
    """
    Returns a feature matrix of degree p of the given data 
    """
    return x[:, None] ** np.arange(int(noIntercept), p+1)

def MSE(target : np.array, pred : np.array) -> float:
    """
    Computes the MSE from the given prediction and target
    """
    return np.average(np.power(target - pred, 2))

def R2(target : np.array, pred : np.array) -> float:
    """
    Computes the R2 from the given prediction and target
    """
    ybar = np.average(target)
    denom = np.sum(np.power(target - ybar, 2))
    
    if denom == 0: return 0.0
    return 1 - np.sum(np.pow(target - pred, 2)) / denom

def testFit(xTest : np.array, yTest : np.array, beta : np.array) -> tuple[float, float]:
    """
    Returns the MSE and R2 of the given input model and data 
    """
    pred = xTest @ beta
    return MSE(yTest, pred), R2(yTest, pred)


def theta_analytic_OLS(X : np.array, y : np.array) -> np.array:
    return np.linalg.pinv(X.T @ X) @ X.T @ y
    #np.linalg.pinv used to ensure numerical stability

def evaluate_OLS_analytic(X_train : np.array, y_train : np.array, X_test : np.array, y_test : np.array) -> tuple[float, np.array]:
    """
    Computes and returns the MSE of OLS closed form solution 
    """        
    theta = theta_analytic_OLS(X_train, y_train)
    y_prediction = X_test @ theta
    mse = MSE(y_prediction, y_test)

    return mse, y_prediction

def theta_analytic_Ridge(X : np.array, y : np.array, lambd = 0.05) -> np.array:
    n_features = X.shape[1]
    return np.linalg.pinv(X.T @ X + lambd * np.identity(n_features)) @ X.T @ y

class GradientDescent:
    """
    Keeps and updates the parameters optimized by gradient descent. Runs the training loop
    """
    def __init__(self, n_features : int, noIntercept : bool = False, stochastichGD : bool = False, logging : bool = False) -> None:
        self.n_features = n_features
        self.noIntercept = noIntercept
        self.stochastichGD = stochastichGD

        self.logging = logging

        # Initialize weights for gradient descent
        self.theta = np.random.rand(n_features + int(not noIntercept))

    def setOptimizer(self, optimizer : Optimizers) -> None:
        self.optimizer = optimizer

    def setGradient(self, gradient : Gradients) -> None:
        self.gradient = gradient

    def forward(self, x_train : np.array, y_train : np.array) -> np.array:
        gradient = self.gradient(self.theta, x_train, y_train)
        self.theta = self.optimizer(self.theta, gradient)
    
    def predict(self, X_test : np.array) -> np.array:
        return X_test @ self.theta
    
    def evaluate(self, X_test : np.array, y_test : np.array) -> float:
        prediction = self.predict(X_test)
        return MSE(prediction, y_test), prediction

    def train(self, X_train : np.array, y_train : np.array, X_test : np.array, y_test : np.array, epoch : int = 100):
        learningRate = self.optimizer.learningRate

        # For early stopping
        minChange = min(learningRate/1000, 0.001)  # sets the minimum change in MSE before early stopping
        numDiffs = 10 # sets number of consecutive steps that must change less than this minimum change
        mseDiffs = np.ones(numDiffs)  # keeps track of MSE diferences for early stopping 

        thetas = []
        MSEs = np.zeros(epoch)

        for t in range(epoch):
            if (self.stochastichGD): # stochastic gradient descent 
                x_train_re, y_train_re = resample(X_train, y_train, random_state=1)
                y_train_re = y_train_re.flatten()
                self.forward(featureMat(x_train_re, self.n_features, noIntercept=self.noIntercept), y_train_re)
            else: # normal gradient descent
                self.forward(X_train, y_train)

            thetas.append(self.theta)

            MSEs[t], prediction = self.evaluate(X_test, y_test)

            # Early stopping
            mseDiffs[t%numDiffs] = abs(MSEs[t]-MSEs[t-1]) # update the stored MSEs
            if (np.all(mseDiffs < minChange)):
                break

        if (self.logging):
            print(f"Training complete!\nTrained for {t} epoch with learning rate {learningRate},\
                the model used {self.n_features} features,\
                optimizer {self.optimizer} and gradient {self.gradient}.\n\
                The best MSE was {MSEs.min():.3f} and was achived after {np.where(MSEs == MSEs.min())[0][0]} epochs.\
                The final MSE was {MSEs[-1]:.3f}.")
        
        return self.theta, MSEs, t

    def evaluation_function(self):
        """
        Used to compute boostrap
        """
        def evaluate_model(x_train, y_train, x_test, y_test) -> float:
            
            X_train = featureMat(x_train, self.n_features, noIntercept=self.noIntercept)
            X_test = featureMat(x_test, self.n_features, noIntercept=self.noIntercept)
            self.train(X_train, y_train, X_test, y_test)
            score, predictions = self.evaluate(X_test, y_test)
            return score, predictions
        return evaluate_model