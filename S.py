# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:47:15 2022

@author: khale
"""

import numpy as np
from scipy import optimize 
from scipy.interpolate import CubicSpline as CS
import matplotlib.pyplot as plt

L1=3
L2=3
X=np.array([2,1])


def R(TH):
    return np.array([L1*np.cos(TH[0])+L2*np.cos(TH[0]+TH[1])-X[0],L1*np.sin(TH[0])+L2*np.sin(TH[0]+TH[1])-X[1]])



def R_norm(TH):
    return (L1*np.cos(TH[0])+L2*np.cos(TH[0]+TH[1])-X[0])**2 + (L1*np.sin(TH[0])+L2*np.sin(TH[0]+TH[1])-X[1])**2


   