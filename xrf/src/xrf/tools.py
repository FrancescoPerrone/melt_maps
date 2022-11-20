#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Francesco Perrone
@email : francesco.perrone@glasgow.ac.uk  

This file is TEMPORARY.
DOCUMENTATION needs improvement.

This file contains a collection of python implementations for XRF data
analysis.

"""

import err
import os

from sklearn import cluster
from scipy.spatial import distance
import sklearn.datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

import tkinter as tk
from tkinter import filedialog as fd
# prevents an empty tkinter window from appearing
tk.Tk().withdraw() 

def init_folder(flag = False):
    """ Select a folder using tkinter

    Parameters:
    flag (Boolean): temporarily set to False. Can be ignored at this stage.

    Returns:
    str:return a string which represents the Path chosen.

    # check if exp_folder entry exists
    # ###### if not create the entry/file
    # ###### if yes exit()
   """
    # checks the setting file is set with a init.csv
    # should be made into tow different steps but it is okay just now.
    # print(os.getcwd())
    #print(os.path.join(os.getcwd(), 'settings', 'init.csv'))
    #os.path.exists(os.path.join(os.getcwd(), 'settings', 'init.csv'))
    #error check in if should go in a separated file.
    exp_folder = str(fd.askdirectory())
    if os.path.isdir(exp_folder): return exp_folder
    else:
        print(err_l.get("NO_FOLDER_FOUND_DES"))
        return err_l.get("NO_FOLDER_FOUND")


def init_file():
    """ Select a file or multiple files using tkinter
    Parameters: n/a
    
    Returns:
    tuple:return a tuple of strings representing file names.

    Tuple items are ordered, unchangeable, and allow duplicate values.
    Tuple items are indexed, the first item has index [0], 
    the second item has index [1] etc.
    """

    root = tk.Tk()
    file_list = fd.askopenfilenames(parent=root, title='* Select a file or files *')
    return file_list


def compute_bic(kmeans,data_points):
    """
    A function to compute the BIC for a given clusters,
    in terms of k.
    Parameters:
    ----------
    kmeans     :  List of clustering object.
                  Just now it needs to be from scikitlearn
    data_points:  a multidimension array of data points
             Just now I left it as nparray but can be
             anything we need. I will authomate the
             conversion process if needed.
    Returns:
    ---------
    BIC value
    """

    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = np.bincount(labels)
    #size of data set
    N, d = data_points.shape

    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(data_points[np.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])

    const_term = 0.5 * m * np.log(N) * (d+1)

    BIC = np.sum([n[i] * np.log(n[i]) -
               n[i] * np.log(N) -
             ((n[i] * d) / 2) * np.log(2*np.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term

    return(BIC)
