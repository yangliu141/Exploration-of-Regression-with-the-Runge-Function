import numpy as np

class Optimizers: 
    class ADAM:
        """
        Implmenets the ADAM optimizer
        """
        def __init__(self, learningRate : float, decay1 : float, decay2 : float, n : int) -> None:
            self.beta1 = decay1
            self.beta2 = decay2
            self.learningRate = learningRate
            self.m = np.zeros(n)
            self.v = np.zeros(n)
            self.epsilon = 1E-8
            self.t = 0

        def __call__(self, theta : np.array, gradient : np.array) -> np.array:
            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            theta = theta - self.learningRate * m_hat / np.sqrt(v_hat + self.epsilon)
            return theta
        
        def __str__(self) -> str: return "ADAM"

    class RMSprop:
        """
        Implmenets the RMSprop optimizer
        """
        def __init__(self, learningRate : float, decay : float, n : int) -> None:
            self.learningRate = learningRate
            self.decay = decay
            self.movingAverage2 = np.zeros(n)
            self.epsilon = 1E-8

        def __call__(self, theta : np.array, gradient : np.array) -> np.array:
            self.movingAverage2 = self.decay*self.movingAverage2 + (1-self.decay) * np.power(gradient, 2)
            theta = theta - self.learningRate * gradient / np.sqrt(self.movingAverage2 + self.epsilon)
            return theta
        
        def __str__(self) -> str: return "RMSprop"

    class ADAgrad:
        """
        Implements the ADAgrad optimizer
        """
        def __init__(self, learningRate : float, n : int) -> None:
            self.learningRate = learningRate
            self.gradSum = np.zeros(n)
            self.epsilon = 1E-8

        def __call__(self, theta : np.array, gradient : np.array) -> np.array:
            self.gradSum = self.gradSum + np.power(gradient, 2)
            lr = self.learningRate / (np.sqrt(self.gradSum)+self.epsilon)
            theta = theta - lr * gradient
            return theta
        
        def __str__(self) -> str: return "ADAgrad"

    class Simple:
        """
        Simple gradient descent with constant learningrate. 
        """
        def __init__(self, learningRate : float) -> None:
            self.learningRate = learningRate

        def __call__(self, theta : np.array, gradient : np.array) -> np.array:
            return theta - self.learningRate * gradient
        
        def __str__(self) -> str: return "No optimizer"

    class Momentum:
        """
        Implements gradient descent with momentom
        """
        def __init__(self, learningRate : float, momentum : float) -> None:
            self.learningRate = learningRate
            self.momentum = momentum
            self.lastTheta = None
        
        def __call__(self, theta : np.array, gradient : np.array) -> np.array:
            self.lastTheta = theta
            return theta - self.learningRate * gradient + self.momentum*(theta - self.lastTheta)
        
        def __str__(self) -> str: return "Momentum"