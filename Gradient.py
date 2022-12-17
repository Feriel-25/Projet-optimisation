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
    """
    

    Parameters
    ----------
    TH0 : Configuration Initialle
    alpha : Le pas du gradient 
    Params : Vecteur contenant L1,L2 et un vecteur numPy X du point desiré
    eps : La precision 
    n_max : Nombre d'iteration Maximale
    IsoV (Optionnel): True : Trace les Isovaleurs a la fin de l'algorithme
           False : Ne Trace rien.
        default: False.
    Disp (Optionnel):   True : Affiche les differents information a travers les calculs
           False : N'affiche rien.
        default: False.

    Returns
    -------
    thn : Solution a la fin du calcul
    converge :  True : L'algorithme a converger vers la solution
           False : L'algorithme a depasser le nombre d'iteration avant max avant de converger vers la solution

    """
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
    """
    Cette algorithm de gradient decend Divise la valeurs du pas sur deux si la fonction minimisé ne 
    decroit pas entre le point n et n+1

    Parameters
    ----------
    TH0 : Configuration Initialle
    alpha : Le pas du gradient 
    Params : Vecteur contenant L1,L2 et un vecteur numPy X du point desiré
    eps : Le precision 
    n_max : Nombre d'iteration Maximale'
    IsoV (Optionnel): True : Trace les Isovaleurs a la fin de l'algorithme
           False : Ne Trace rien.
        default: False.
    Disp (Optionnel):   True : Affiche les differents information a travers les calculs
           False : N'affiche rien.
        default: False.

    Returns
    -------
    thn : Solution a la fin du calcul
    converge :  True : L'algorithme a converger vers la solution
           False : L'algorithme a depasser le nombre d'iteration avant max avant de converger vers la solution

    """
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