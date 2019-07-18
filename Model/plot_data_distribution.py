#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def plot_data_distribution(y_data, labels, title ="Traing Data", figsize=(18,8), rot=0):
    # Based on y_data which represent the vector with all data 
    # plot the distributuion of the different class 
    tempLABELS = np.asarray(labels)
    
    # Count instances of each class represented in the data
    _, counts = np.unique(y_data, return_counts=True)
    
    # Create (pandas) data frame 
    df = pd.DataFrame({'Instances':counts}, index=tempLABELS)
    ax = df.plot.bar(title =title, figsize=figsize, rot=rot, color=list('g'))
    ax.set_xlabel("Exercises", fontsize=12)
    # Plot whole diagram
    plt.show()