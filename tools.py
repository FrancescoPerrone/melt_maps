#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Francesco Perrone
@email : francesco.perrone@glasgow.ac.uk  

This file is TEMPORARY.
DOCUMENTATION needs improvement.

This file contains a collection of python implementations for the melt_maps project

"""

import err
import os
import tkinter
from tkinter import filedialog
# prevents an empty tkinter window from appearing
tkinter.Tk().withdraw() 

def folder_init(flag = False):
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
    exp_folder = str(filedialog.askdirectory())
    if os.path.isdir(exp_folder): return exp_folder
    else:
        print(err.get("NO_FOLDER_FOUND_DES"))
        return err.get("NO_FOLDER_FOUND")