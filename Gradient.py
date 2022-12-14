# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 23:52:25 2022

@author: khale
"""
import numpy as np
import matplotlib.pyplot as plt
from Functions import Grad_Re , R_norm 
from Graphics import isoValeurs


def GradienDecent (TH0,alpha,Params,eps,n_max,IsoV=False, Disp = False):
    Un = 1
    thn = TH0 
    n=0
    if (IsoV == True) : 
        plt.figure()
    while(np.linalg.norm(Un)>eps and n<n_max ) :
        dx = alpha * Grad_Re(thn[0],thn[1],Params)
        thn_1 =  thn - dx
        Un = thn_1 - thn 
        if (IsoV) : 
            isoValeurs (Params,100,100,25,thn,thn_1) 
        thn = thn_1
        n=n+1
        if (Disp) :
            print(f" Iterations : {n} \t alpha = {alpha} \t dX = {np.linalg.norm(dx)}")
    if (n<n_max) : 
        converge = True
        if (Disp) :
            print(f"Minimum trouve apres {n} iterations")
    else : 
        converge = False
        print("Gradient ne converge pas")
    return thn , converge

def GradienDecentAmeliore (TH0,alpha,Params,eps,n_max,IsoV=False, Disp = False):
    Un = 1
    divide = False
    thn = TH0 
    n=0
    alphaInit = alpha
    if (IsoV == True) : 
        plt.figure()
    while(np.linalg.norm(Un)>eps and n<n_max ) :
        dx = alpha * Grad_Re(thn[0],thn[1],Params)
        thn_1 =  thn - dx
        if (R_norm(thn,Params)<R_norm(thn_1,Params)) and not divide  : 
            alpha = alphaInit / 2 
            divide = True
        elif(R_norm(thn,Params)>R_norm(thn_1,Params) and divide):
            alpha = alphaInit
            divide = False
        Un = thn_1 - thn 
        if (IsoV) : 
            isoValeurs (Params,100,100,25,thn,thn_1) 
        thn = thn_1
        n=n+1
        if (Disp) :
            print(f" Iterations : {n} \t alpha = {alpha} \t dX = {np.linalg.norm(dx)}")
    if (n<n_max) : 
        converge = True
        if (Disp) :
            print(f"Minimum trouve apres {n} iterations")
    else : 
        converge = False
        print("Gradient ne converge pas")
    return thn , converge