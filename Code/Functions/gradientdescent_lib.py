import numpy as np
import matplotlib.pyplot as plt
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
    y = 1 / (1 + 25*np.pow(x, 2)) + np.random.normal(0, noise, size=nData)

    return train_test_split(normalize(x.reshape(-1, 1), axis=0, norm='max'), y)

def featureMat(x : np.array, p : int, noIntercept : bool = True) -> np.array:
    """
    Returns a feature matrix of degree p of the given data 
    """
    return x[:, None] ** np.arange(int(noIntercept), p+1)

def MSE(target : np.array, pred : np.array) -> float:
    """
    Computes the MSE from the given prediction and target
    """
    return np.average(np.pow(target - pred, 2))

def R2(target : np.array, pred : np.array) -> float:
    """
    Computes the R2 from the given prediction and target
    """
    ybar = np.average(target)
    denom = np.sum(np.pow(target - ybar, 2))
    
    if denom == 0: return 0.0
    return 1 - np.sum(np.pow(target - pred, 2)) / denom

def testFit(xTest : np.array, yTest : np.array, beta : np.array) -> tuple[float, float]:
    """
    Returns the MSE and R2 of the given input model and data 
    """
    pred = xTest @ beta
    return MSE(yTest, pred), R2(yTest, pred)


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

    def train(self, X_train : np.array, y_train : np.array, X_test : np.array, y_test : np.array, epoch : int):
        learningRate = self.optimizer.learningRate

        # For early stopping
        minChange = min(learningRate/1000, 0.001)  # sets the minimum change in MSE before early stopping
        numDiffs = 10 # sets number of consecutive steps that must change less than this minimum change
        mseDiffs = np.ones(numDiffs)  # keeps track of MSE diferences for early stopping 

        thetas = []
        MSEs = np.zeros(epoch)

        for t in range(epoch):
            if (self.stochastichGD): # stochastic gradient descent 
                x_train_re, y_train_re = resample(X_train, y_train)
                y_train_re = y_train_re.flatten()
                self.forward(featureMat(x_train_re, self.n_features, noIntercept=self.noIntercept), y_train_re)
            else: # normal gradient descent
                self.forward(X_train, y_train)

            thetas.append(self.theta)

            MSEs[t] = self.evaluate(X_test, y_test)

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
        def evaluate_model(x_train, y_train, x_test, y_test) -> float:
            
            X_train = featureMat(x_train, self.n_features, noIntercept=self.noIntercept)
            X_test = featureMat(x_test, self.n_features, noIntercept=self.noIntercept)
            self.train(X_train, y_train)
            score = self.evaluate(X_test, y_test)
            return score
        return evaluate_model



#--------------------------------------------------------------
#---------------------Real code starts here--------------------
#NOTE: Vet ikke om dette fungerer lengre.
'''
def main(): 
    #For å kjøre koden bruker du denne (versjon 1): 

    np.random.seed(10)

    x_train, x_test, y_train, y_test = generateData(100)
    x_train = x_train.flatten(); x_test = x_test.flatten()

    epoch = 100
    learningRate = 0.05
    minChange = min(learningRate/1000, 0.001)
    noIntercept = False

    n_features = 4

    mses = np.zeros(epoch)
    R2s = np.zeros(epoch)
    thetas = []

    gd = GradientDescent(n_features+int(not noIntercept), momentum=0)
    #gd.setOptimizer(Optimizers.RMSprop(learningRate, 0.99, n_features+int(not noIntercept)))
    #gd.setOptimizer(Optimizers.ADAgrad(learningRate, n_features+int(not noIntercept)))
    gd.setOptimizer(Optimizers.ADAM(learningRate, 0.9, 0.999, n_features+int(not noIntercept)))
    g = Gradients.Ridge(0.01)
    gd.setGradient(g)

    x_1 = np.linspace(-1, 1, 100)
    y_1 = 1 / (1 + 25*np.pow(x_1, 2)) + np.random.normal(0, 0.1, size=100)

    x_test = featureMat(x_test, n_features, noIntercept=noIntercept)
    # Gradient descent loop
    for t in range(epoch):
        x_train_re, y_train_re = resample(x_train, y_train)
        x_train_re = featureMat(x_train_re, n_features, noIntercept=noIntercept)
        y_train_re = y_train_re.flatten()

        theta = gd.forward(featureMat(x_train, n_features, noIntercept=noIntercept), y_train)

        thetas.append(theta)
        mses[t], R2s[t] = testFit(x_test, y_test, theta)

        # Early stopping
        if (abs(mses[t]-mses[t-1]) < minChange): break


    print(t)
    plt.scatter(x_train, featureMat(x_train, n_features, noIntercept=noIntercept)@thetas[-1], label=t)
    plt.scatter(x_train, y_train)
    plt.legend()
    plt.show()

    plt.plot(range(epoch), mses, label="MSEs")
    #plt.plot(range(epoch), R2s, label="R2")
    #plt.plot(range(len(thetas)), thetas)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.legend()
    plt.show()






    # eller du kan kjøre den med denne (versjon 2)

    np.random.seed(1)

    x_train, x_test, y_train, y_test = generateData(100)
    x_train = x_train.flatten(); x_test = x_test.flatten()

    epoch = 100
    learningRate = 0.05
    minChange = min(learningRate/1000, 0.001)
    noIntercept = False

    n_featuresList = [2, 4, 6]

    logs = []

    for n_features in n_featuresList:
        print("Numb features: ", n_features)
        # Define the search space for this number of features 
        gradients = [Gradients.OLS(), 
                Gradients.Ridge(0.01), 
                Gradients.Lasso(0.01)]
        optimizers = [Optimizers.ADAM(learningRate, 0.9, 0.999, n_features+int(not noIntercept)),
                    Optimizers.RMSprop(learningRate, 0.99, n_features+int(not noIntercept)),
                    Optimizers.ADAgrad(learningRate, n_features+int(not noIntercept))]
        
        combinations = [(grad, opt) for grad in gradients for opt in optimizers]
        
        gd = GradientDescent(n_features+int(not noIntercept), momentum=0)
        for comb in combinations:
            gd.setOptimizer(comb[1])
            gd.setGradient(comb[0])

            x_test_feat = featureMat(x_test, n_features, noIntercept=noIntercept)

            mses = np.zeros(epoch)
            R2s = np.zeros(epoch)
            thetas = []
            numDiffs = 10
            mseDiffs = np.ones(numDiffs)

            best = None
            gotBest = False

            # Gradient descent loop
            for t in range(epoch):
                theta = gd.forward(featureMat(x_train, n_features, noIntercept=noIntercept), y_train)

                thetas.append(theta)
                mses[t], R2s[t] = testFit(x_test_feat, y_test, theta)

                # Early stopping
                mseDiffs[t%numDiffs] = abs(mses[t]-mses[t-1])

                if (np.all(mseDiffs < minChange)):
                    break
            logs.append(f"Opt: {str(comb[1])}      grad: {str(comb[0])}      epoch: {t:3}      mse: {mses[t]:.4f}")


    with open("out.txt", "w") as outfile:
        outfile.write("\n".join(logs))

if __name__ == "__main__":
    #runs code if not imported
    main()

#Dette skal kanskje kjøre?
'''