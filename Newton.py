# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 08:59:21 2022
@author: feriel
"""
import numpy as np
from scipy import optimize 
from scipy.interpolate import CubicSpline as CS
import matplotlib.pyplot as plt
from optimisation_projet import Grad_Re ,  Hess_Re , F

print('Hana f newton')

th1n = -0.5
th2n = 2
TH=np.array([th1n,th2n])
n = 1
L1=3
L2=3


X=np.array([2,1])



print(Newton(0.1,TH,X,1000))
