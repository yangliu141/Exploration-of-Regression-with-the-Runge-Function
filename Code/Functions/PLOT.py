import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np
from matplotlib.ticker import MaxNLocator

def plot(
        nGraphs: int,
        x_axis_array : list,
        y_features : list,
        y_feature_label = None,
        foldername: str = 'figures',
        figurename: str = 'figure1',
        x_label: str = 'x',
        y_label: str = 'y',
        title: str = '',
        x_integer_entries: bool = False,
        y_integer_entries: bool = False,
        save : bool = False,
        scatter : list = None,
        multiX : bool = False,
        diffColor : bool = True
):
    '''
    NOTE: In a notebook, run "%matplotlib inline" in your notebook before this function if you want plot displayed inline.
    
    This function takes the values above and makes a plot formated for reports in latex.
    It saves the plot as a pdf as [figurename].pdf in folder [foldername], and plots it inline in a notebook.

    y_features is an array where each entry is a list of y values to plot. Entry i of the list should correspond
    to entry i of x_axis array.

    NOTE: 
    len(x_axis_array) = len(y_features[i]) for all indexes i, 
    len(y_features) = len(y_features_label) = n
    
    '''

    # default scatter to of for all graphs 
    if scatter == None:
        scatter = [False] * nGraphs

    mpl.rcParams.update({
        "font.family": "serif",    # match LaTeX document
        "font.size": 10,           # document font size
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    if y_feature_label == None:
        y_feature_label = list(range(nGraphs))

    # Folder to save figures
    FIGURE_FOLDER = foldername
    os.makedirs(FIGURE_FOLDER, exist_ok=True)

    # Figure size & line width
    COLUMNWIDTH_PT = 246.0           # LaTeX \columnwidth
    INCHES_PER_PT = 1/72.27
    FIG_WIDTH = COLUMNWIDTH_PT * INCHES_PER_PT
    FIG_HEIGHT = FIG_WIDTH * 0.6      # adjust aspect ratio
    STANDARD_LINEWIDTH = 1.5

    # -------------------- Plotting function --------------------

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    for i in range(nGraphs):
        x = x_axis_array[i] if multiX else x_axis_array
        if scatter[i]:
            ax.scatter(x, y_features[i], label=y_feature_label[i], linewidth=STANDARD_LINEWIDTH)
        else:
            if diffColor:
                ax.plot(x, y_features[i], label=y_feature_label[i], linewidth=STANDARD_LINEWIDTH)
            else:
                ax.plot(x, y_features[i], label=y_feature_label[i], linewidth=STANDARD_LINEWIDTH, color="orange")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if x_integer_entries:
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    if y_integer_entries:
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.legend()
    
    # Optional: adjust margins if labels clip
    plt.subplots_adjust(left=0.18, right=0.95, top=0.9, bottom=0.22)
    
    
    # -------------------- Display and save LaTeX-quality PDF --------------------
    if save:
        plt.savefig(f"{FIGURE_FOLDER}/{figurename}.pdf", format='pdf')
    plt.show() 
    plt.close(fig)  # free memory

