#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 20:08:28 2021

@author: henryrodriguez
"""

import matplotlib.pyplot as plt
import numpy as np
import embedd_blocks as em_blocks
import time
import pegasos as pegasos

from sklearn.datasets import load_svmlight_files
#%%
import pandas as pd

from sklearn.datasets import load_files
#%%
data_train = pd.read_csv("datasets/mnistbinary.train", header=None, delim_whitespace=True, skiprows=[0])
#%%
df_X_train = data_train.drop([0], axis=1)
df_y_train = data_train[0]

X_train = df_X_train.to_numpy()
y_train = df_y_train.to_numpy()

#digit_pixels = X_train[11].reshape(28,28)
#plt.imshow(digit_pixels)
#%%
X_train, y_train, X_test, y_test = load_svmlight_files(("a9a/a9a", "a9a/a9a.t"))
#%%
m_list = [100, 1000, 3000, 5000]
lambda_list = [1, 0.1, 0.01 , 1e-03, 1e-04, 1e-05, 1e-06]

em_blocks.exec(X_train, y_train, m_list, lambda_list)