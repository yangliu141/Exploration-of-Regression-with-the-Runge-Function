import numpy as np
import pandas as pd  # used to create Latex tables 
from sklearn.linear_model import LinearRegression, Ridge, Lasso   # used for benchmarking

from Functions.gradientdescent_lib import generateData, featureMat, GradientDescent, MSE
from Functions.gradients import Gradients
from Functions.optimizers import Optimizers

class RunAllExperiments:
    def __init__(self) -> None:
        self.logs = None

    def run(self, logging : bool) -> None:
        """
        Computes the test MSE and number of epochs for convergence for the different model complexities, optimizers and gradients
        """
        np.random.seed(1)

        # define the data
        self.x_train, self.x_test, self.y_train, self.y_test = generateData(100, 0.1)
        self.x_train = self.x_train.flatten(); self.x_test = self.x_test.flatten()

        # standard parameters
        epoch = 100
        learningRate = 0.05
        self.noIntercept = False

        self.n_featuresList = [2, 4, 6] # searchspace for model complexity 

        logs = []
        logs.append("Optimizer                Gradient        Number epoch      Last MSE")

        self.msesTables = []
        self.convergenceTables = []

        # loops through the different model complexities 
        for n_features in self.n_featuresList:
            logs.append(f"Numb features: {n_features}")

            # Define the search space for this number of features 
            featuresWithIntercept = n_features+int(not self.noIntercept)
            gradients = [Gradients.OLS(), 
                        Gradients.Ridge(0.01), 
                        Gradients.Lasso(0.01)]
            optimizers = [Optimizers.Simple(learningRate),
                        Optimizers.ADAM(learningRate, 0.9, 0.999, featuresWithIntercept),
                        Optimizers.RMSprop(learningRate, 0.99, featuresWithIntercept),
                        Optimizers.ADAgrad(learningRate, featuresWithIntercept),
                        Optimizers.Momentum(learningRate, 0.9)]
            
            # get all pairs of gradient-optimizer combinations 
            combinations = [(grad, opt) for grad in gradients for opt in optimizers]

            allMses = np.zeros((len(optimizers), len(gradients)))
            finalEpoch = np.zeros((len(optimizers), len(gradients)))

            x_test_feat = featureMat(self.x_test, n_features, noIntercept=self.noIntercept)
            x_train_feat = featureMat(self.x_train, n_features, noIntercept=self.noIntercept)
            
            counter = 0
            for comb in combinations:
                gd = GradientDescent(n_features, logging=logging)
                gd.setOptimizer(comb[1])
                gd.setGradient(comb[0])

                theta, mses, numberEpoch = gd.train(x_train_feat, self.y_train, x_test_feat, self.y_test, epoch)

                logs.append(f"Opt: {str(comb[1]):14}      grad: {str(comb[0])}      epoch: {numberEpoch:3}      mse: {mses[numberEpoch]:.4f}")
                
                allMses[counter%len(optimizers)][int(counter / len(optimizers))] = mses[numberEpoch]
                finalEpoch[counter%len(optimizers)][int(counter / len(optimizers))] = numberEpoch
                
                counter += 1
            
            self.convergenceTables.append(pd.DataFrame(finalEpoch, index=optimizers, columns=gradients))
            self.msesTables.append(pd.DataFrame(allMses, index=optimizers, columns=gradients))
        self.logs = logs


    def getBenchMarkMSE(self) -> np.array:
        """
        Computes the test MSE for the closed form solutions of OLS, Ridge and Lasso  
        """
        benchMarkMSEs = np.zeros((3, len(self.n_featuresList)))

        for i, n_features in enumerate(self.n_featuresList):
            x_test_feat = featureMat(self.x_test, n_features, noIntercept=self.noIntercept)
            x_trian_feat = featureMat(self.x_train, n_features, noIntercept=self.noIntercept)

            ols = LinearRegression(fit_intercept=True)
            ols.fit(x_trian_feat, self.y_train)
            benchMarkMSEs[0][i] = MSE(ols.predict(x_test_feat), self.y_test)


            ridge = Ridge(alpha=0.01)
            ridge.fit(x_trian_feat, self.y_train)
            benchMarkMSEs[1][i] = MSE(ridge.predict(x_test_feat), self.y_test)

            lasso = Lasso(alpha=0.01)
            lasso.fit(x_trian_feat, self.y_train)
            benchMarkMSEs[2][i] = MSE(lasso.predict(x_test_feat), self.y_test)
        return benchMarkMSEs
        

    def exportLogs(self, file_path : str):
        with open(file_path, "w") as outfile:
            outfile.write("\n".join(self.logs))
        print("Wrote logs to file ", file_path)


    def exportLogsLatex(self, type : str, file_path : str, writeType : str = "w") -> None:
        """
        Exports the MSE [MSE], convergence [c] and benchMark [bm] logs to latex format.
        """
        if type.lower() == "mse":
            with open(file_path, writeType) as f:
                f.write("\\newcommand{\\results}{\n")
                for i, table in enumerate(self.msesTables, start=0):
                    f.write(f"% ---- {self.n_featuresList[i]} features ----\n")
                    f.write(f"\\begin{{table}}[H]\n\\centering\n\\caption{{Final test MSE for gradient descent. Showing gradient-optimizer combinations with {self.n_featuresList[i]} features.}}\n"+"\\label{tab:training_mse"+str(self.n_featuresList[i])+"}")
                    f.write(table.to_latex(float_format="%.4f"))
                    f.write("\\end{table}\n\n")
                f.write("}\n")

        if type.lower() == "c":
            with open(file_path, writeType) as f:
                for i, table in enumerate(self.convergenceTables, start=0):
                    f.write(f"\n% ---- {self.n_featuresList[i]} features ----\n")
                    f.write(f"\\begin{{table}}[H]\n\\centering\n\\caption{{Number of epoch need for convergence for gradient-optimizer combinations with {self.n_featuresList[i]} features.}}\n"+"\\label{tab:final_epoch"+str(self.n_featuresList[i])+"}")
                    f.write(table.to_latex(float_format="%d"))
                    f.write("\\end{table}\n")

        if type.lower() == "bm":
            df = pd.DataFrame(self.getBenchMarkMSE(), index=["OLS", "Ridge", "Lasso"], columns=self.n_featuresList)

            with open(file_path, writeType) as f:
                f.write("\\newcommand{\\benchMark}{\n")
                f.write(f"% ---- Benchmark values ----\n")
                f.write(f"\\begin{{table}}[H]\n\\centering\n\\caption{{Test MSEs from closed form solutions. Gradient plotted against number of features.}}\n"+"\\label{tab:benchMarkMSE}")
                f.write(df.to_latex(float_format="%.4f"))
                f.write("\\end{table}\n")
                f.write("}")

        print("Wrote tabels to file ", file_path)




