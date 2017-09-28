# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 20:57:17 2017

@author: User
"""

import numpy as np
import time
import scipy.io as sio
import sklearn.metrics.pairwise as kernels
from scipy.linalg import inv
from scipy.sparse import csgraph as cg
from IPython import embed
from SVC_init import *

data_name='data/ring'
input = load_data(data_name)
print("with data",data_name)
support = "GP"
hyperparams = [100*np.ones((input.shape[0],1)), 1, 10]
#hyperparams = {'ker': 'rbf', 'arg': 0.5, 'C': 0.5 }

model = supportmodel(input,support,hyperparams)

options = {'hierarchical': False, 'K': 4, 'epsilon': 0.05, 'R1': 0, 'R2': 0}
labmodel = labeling(model,"T-MSC", options)

embed()
