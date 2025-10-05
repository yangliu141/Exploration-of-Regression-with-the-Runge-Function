import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import normalize


def generateData(nData : int, noise_std: float = 0.1): 
    x = np.linspace(-1, 1, nData)
    y = 1 / (1 + 25*np.pow(x, 2)) + np.random.normal(0, noise_std, size=nData)

    return train_test_split(normalize(x.reshape(-1, 1), axis=0, norm='max'), y)

def featureMat(x : np.array, p : int, noIntercept : bool = True) -> np.array:
    return x[:, None] ** np.arange(int(noIntercept), p+1)

def MSE(target : np.array, pred : np.array) -> float:
    return np.average(np.pow(target - pred, 2))

def R2(target : np.array, pred : np.array) -> float:
    ybar = np.average(target)
    denom = np.sum(np.pow(target - ybar, 2))
    
    if denom == 0: return 0.0
    return 1 - np.sum(np.pow(target - pred, 2)) / denom

def testFit(xTest : np.array, yTest : np.array, beta : np.array) -> tuple[float, float]:
    pred = xTest @ beta
    return MSE(yTest, pred), R2(yTest, pred)

class Optimizers: 
    class ADAM:
        """
        Implmenets the ADAM optimizer
        """
        def __init__(self, learningRate, decay1, decay2, n):
            self.beta1 = decay1
            self.beta2 = decay2
            self.lr = learningRate
            self.m = np.zeros(n)
            self.v = np.zeros(n)
            self.epsilon = 1E-8
            self.t = 0

        def __call__(self, theta, gradient):
            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            theta = theta - self.lr * m_hat / np.sqrt(v_hat + self.epsilon)
            return theta
        
        def __str__(self): return "ADAM   "

    class RMSprop:
        """
        Implmenets the RMSprop optimizer
        """
        def __init__(self, learningRate, decay, n):
            self.lr = learningRate
            self.decay = decay
            self.movingAverage2 = np.zeros(n)
            self.epsilon = 1E-8

        def __call__(self, theta, gradient):
            self.movingAverage2 = self.decay*self.movingAverage2 + (1-self.decay) * np.pow(gradient, 2)
            theta = theta - self.lr * gradient / np.sqrt(self.movingAverage2 + self.epsilon)
            return theta
        
        def __str__(self): return "RMSprop"

    class ADAgrad:
        """
        Implements the ADAgrad optimizer
        """
        def __init__(self, learningRate, n):
            self.learningRate = learningRate
            self.gradSum = np.zeros(n)
            self.epsilon = 1E-8

        def __call__(self, theta, gradient):
            self.gradSum = self.gradSum + np.pow(gradient, 2)
            lr = self.learningRate / (np.sqrt(self.gradSum)+self.epsilon)
            theta = theta - lr * gradient
            return theta
        
        def __str__(self): return "ADAgrad"

    class Simple:
        """
        Simple gradient descent with constant learningrate. 
        """
        def __init__(self, learningRate):
            self.learningRate = learningRate

        def __call__(self, theta, gradient):
            return theta - self.learningRate * gradient
        def __str__(self): return "No optimizer"

    class Momentum:
        """
        Implements gradient descent with momentom
        """
        def __init__(self, learningRate, momentum):
            self.learningRate = learningRate
            self.momentum = momentum
            self.lastTheta = None
        
        def __call__(self, theta, gradient):
            self.lastTheta = theta
            return theta - self.learningRate * gradient + self.momentum*(theta - self.lastTheta)
        
        def __str__(self): return "Momentum"

class Gradients:
    class OLS:
        def __call__(self, theta, X, y):
            return 2/X.shape[0] * (X.T @ X @ theta - X.T @ y)
        def __str__(self): return "OLS  "
        
    class Ridge:
        def __init__(self, l):
            self.l = l
        def __call__(self, theta, X, y):
            return 2/X.shape[0] * (X.T @ X @ theta - X.T @ y) + 2*self.l*theta
        def __str__(self): return "Ridge"
    
    class Lasso:
        def __init__(self, l):
            self.l = l
        def __call__(self, theta, X, y):
            return 2/X.shape[0] * (X.T @ X @ theta - X.T @ y) + self.l*np.sign(theta)
        def __str__(self): return "Lasso"

class GradientDescent:
    """
    Keeps and updates the parameters optimized by gradient descent.
    """
    def __init__(self, n_features, seed = 0):
        self.n_features = n_features
        if seed != 0: np.random.seed(seed)

        # Initialize weights for gradient descent
        self.theta = np.random.rand(n_features)

    def setOptimizer(self, optimizer):
        self.optimizer = optimizer

    def setGradient(self, gradient):
        self.gradient = gradient

    def forward(self, x_train, y_train):
        gradient = self.gradient(self.theta, x_train, y_train)
        self.theta = self.optimizer(self.theta, gradient)
        return self.theta
    

def trainingLoop(epoch : int, learningRate : float, n_features : int):
    np.random.seed(1)

    x_train, x_test, y_train, y_test = generateData(100)
    x_train = x_train.flatten(); x_test = x_test.flatten()

    minChange = min(learningRate/1000, 0.001)
    noIntercept = False

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



#--------------------------------------------------------------
#---------------------Real code starts here--------------------
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

#trainingLoop(100, 0.05, 4)
#Dette skal kanskje kjøre?
