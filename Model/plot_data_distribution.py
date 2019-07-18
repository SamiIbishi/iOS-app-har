#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_data_distribution(y_data, labels, title ="Traing Data", figsize=(18,8), rot=0):
    # Based on y_data which represent the vector with all data 
    # plot the distributuion of the different class 
    tempLABELS = np.asarray(labels)
    _, counts = np.unique(y_data, return_counts=True)

    df = pd.DataFrame({'Instances':counts}, index=tempLABELS)
    ax = df.plot.bar(title =title, figsize=figsize, rot=rot)
    ax.set_xlabel("Exercises", fontsize=12)
    
    plt.show()