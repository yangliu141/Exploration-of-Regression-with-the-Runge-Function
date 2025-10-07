import numpy as np
import pandas as pd  # used to create Latex tables 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import matplotlib as mpl
from sklearn.linear_model import LinearRegression, Ridge, Lasso   # used for benchmarking

from Functions.gradientdescent_lib import generateData, featureMat, GradientDescent, MSE
from Functions.gradients import Gradients
from Functions.optimizers import Optimizers

class RunAllExperiments:
    """
    Runs all experiments of the different gradient, optimizers and number of features, computes the MSE and saves them to Latex tables. 
    """
    def __init__(self) -> None:
        self.logs = None

    def run(self, logging : bool) -> None:
        """
        Computes the test MSE and number of epochs for convergence for the different model complexities, optimizers and gradients
        """

        # define the data
        self.x_train, self.x_test, self.y_train, self.y_test = generateData(100, 0.1)
        self.x_train = self.x_train.flatten(); self.x_test = self.x_test.flatten()

        # standard parameters
        epoch = 100
        learningRate = 0.05
        self.noIntercept = False

        self.n_featuresList = range(0, 16) # searchspace for model complexity 

        logs = []
        logs.append("Optimizer                Gradient        Number epoch      Last MSE")

        self.msesTables = []
        self.convergenceTables = []

        self.allFeatureMSE = []
        self.allFeatureR2 = []

        # loops through the different model complexities 
        for n_features in self.n_featuresList:
            logs.append(f"Numb features: {n_features}")

            # Define the search space for this number of features 
            featuresWithIntercept = n_features+int(not self.noIntercept)
            self.gradients = [Gradients.OLS(), 
                              Gradients.Ridge(0.01), 
                              Gradients.Lasso(0.01)]
            self.optimizers = [Optimizers.Simple(learningRate),
                               Optimizers.ADAM(learningRate, 0.9, 0.999, featuresWithIntercept),
                               Optimizers.RMSprop(learningRate, 0.99, featuresWithIntercept),
                               Optimizers.ADAgrad(learningRate, featuresWithIntercept),
                               Optimizers.Momentum(learningRate, 0.9, featuresWithIntercept)]
            
            # get all pairs of gradient-optimizer combinations 
            combinations = [(grad, opt) for grad in self.gradients for opt in self.optimizers]

            allMses = np.zeros((len(self.optimizers), len(self.gradients)))
            allR2 = np.zeros((len(self.optimizers), len(self.gradients)))
            finalEpoch = np.zeros((len(self.optimizers), len(self.gradients)))

            x_test_feat = featureMat(self.x_test, n_features, noIntercept=self.noIntercept)
            x_train_feat = featureMat(self.x_train, n_features, noIntercept=self.noIntercept)
            
            counter = 0
            for comb in combinations:
                gd = GradientDescent(n_features, logging=logging)
                gd.setOptimizer(comb[1])
                gd.setGradient(comb[0])

                theta, mses, R2, numberEpoch = gd.train(x_train_feat, self.y_train, x_test_feat, self.y_test, epoch)

                logs.append(f"Opt: {str(comb[1]):14}      grad: {str(comb[0])}      epoch: {numberEpoch:3}      mse: {mses[numberEpoch]:.4f}")
                
                allMses[counter%len(self.optimizers)][int(counter / len(self.optimizers))] = mses[numberEpoch]
                allR2[counter%len(self.optimizers)][int(counter / len(self.optimizers))] = R2[numberEpoch]
                finalEpoch[counter%len(self.optimizers)][int(counter / len(self.optimizers))] = numberEpoch
                
                counter += 1

            self.allFeatureMSE.append(allMses)
            self.allFeatureR2.append(allR2)

            self.convergenceTables.append(pd.DataFrame(finalEpoch, index=self.optimizers, columns=self.gradients))
            self.msesTables.append(pd.DataFrame(allMses, index=self.optimizers, columns=self.gradients))
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
        # Save the MSE values as a Latex table to file
        if type.lower() == "mse":
            with open(file_path, writeType) as f:
                f.write("\\newcommand{\\results}{\n")
                for i, table in enumerate(self.msesTables, start=0):
                    f.write(f"% ---- {self.n_featuresList[i]} features ----\n")
                    f.write(f"\\begin{{table}}[H]\n\\centering\n\\caption{{Final test MSE for gradient descent. Showing gradient-optimizer combinations with {self.n_featuresList[i]} features.}}\n"+"\\label{tab:training_mse"+str(self.n_featuresList[i])+"}")
                    f.write(table.to_latex(float_format="%.4f"))
                    f.write("\\end{table}\n\n")
                f.write("}\n")

        # Save the number of epoch run for each experiments as a Latex table to file
        if type.lower() == "c":
            with open(file_path, writeType) as f:
                for i, table in enumerate(self.convergenceTables, start=0):
                    f.write(f"\n% ---- {self.n_featuresList[i]} features ----\n")
                    f.write(f"\\begin{{table}}[H]\n\\centering\n\\caption{{Number of epoch need for convergence for gradient-optimizer combinations with {self.n_featuresList[i]} features.}}\n"+"\\label{tab:final_epoch"+str(self.n_featuresList[i])+"}")
                    f.write(table.to_latex(float_format="%d"))
                    f.write("\\end{table}\n")

        # Saves as a Latex table to file, the benchmark values from the closed form solution of OLS and Ridge and the optimal values of Lasso
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

    def createHeatMap(self, filename):
        mseTransposed = np.array(self.allFeatureMSE).transpose(2, 1, 0)
        R2Transposed = np.array(self.allFeatureR2).transpose(2, 1, 0)

        mpl.rcParams.update({
            "font.family": "serif",    # match LaTeX document
            "font.size": 10,           # document font size
            "axes.labelsize": 10,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })

        COLUMNWIDTH_PT = 246.0           # LaTeX \columnwidth
        INCHES_PER_PT = 1/72.27
        FIG_WIDTH = COLUMNWIDTH_PT * INCHES_PER_PT
        FIG_HEIGHT = FIG_WIDTH * 0.6      # adjust aspect ratio
        STANDARD_LINEWIDTH = 1.5

        for i, (mse, r2) in enumerate(zip(mseTransposed, R2Transposed)):
            fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH*2, FIG_HEIGHT))

            axes[0].imshow(np.array(mse))
            axes[1].imshow(np.array(r2))

            # MSE plot
            im0 = axes[0].imshow(mse, aspect='auto', cmap='viridis', origin='lower', norm=LogNorm(vmin=mse.min(), vmax=mse.max()))
            axes[0].set_title(f"MSE")
            axes[0].set_yticks(np.arange(len(self.optimizers)))
            axes[0].set_yticklabels(self.optimizers)
            axes[0].set_ylabel("Optimizers")
            axes[0].set_xlabel("Polynimal degree")

            # R2 plot
            im1 = axes[1].imshow(r2, aspect='auto', cmap='plasma', origin='lower', norm=SymLogNorm(linthresh=0.01, linscale=1, vmin=r2.min(), vmax=r2.max()))
            axes[1].set_title(f"R²")
            axes[1].set_yticks(np.arange(len(self.optimizers)))
            axes[1].set_yticklabels(self.optimizers)
            axes[1].set_ylabel("Optimizers")  
            axes[1].set_xlabel("Polynimal degree")

            # Colorbars
            cbar0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            cbar1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            ticks = np.logspace(np.log10(mse.min()), np.log10(mse.max()), num=6)  # 6 ticks
            cbar0.set_ticks(ticks)
            cbar0.set_ticklabels([f"{t:.2f}" for t in ticks]) 

            plt.tight_layout()
            fig.savefig(filename+"HeatmapMSE"+str(self.gradients[i])+".pdf", dpi=300, bbox_inches='tight', format='pdf')
            plt.show()
            plt.close(fig)