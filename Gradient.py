# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 23:52:25 2022

@author: khale
"""
import numpy as np
import matplotlib.pyplot as plt
from Functions import Grad_Re , R_norm 
from Graphics import isoValeurs


def GradienDecent (TH0,alpha,Params,eps,n_max):
    Un = 1
    divide = False
    thn = TH0 
    n=0
    aplha_init = alpha
    plt.figure()
    while(np.linalg.norm(Un)>eps and n<n_max ) :
        thn_1 =  thn - alpha * Grad_Re(thn[0],thn[1],Params)
        if (R_norm(thn,Params)<R_norm(thn_1,Params)) and not divide  : 
            alpha = aplha_init / 2 
            divide = True
        elif(R_norm(thn,Params)>R_norm(thn_1,Params) and divide):
            alpha = aplha_init
            divide = False
        Un = thn_1 - thn 
        isoValeurs (Params,100,100,25,thn,thn_1) 
        thn = thn_1
        n=n+1
    if (n<n_max) : converge = True
    else : converge = False
    return thn , converge