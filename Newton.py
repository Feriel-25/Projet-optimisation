# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 23:54:31 2022

@author: khale
"""
import numpy as np
from scipy import optimize 
from Functions import Newton_eq 

def Newton(TH0,Params,eps,nmax):
    global grad 
    global Hes
    mod=1
    th1n=TH0[0]
    th2n=TH0[1]
    n=0
    q = [np.pi/4,np.pi/4]
    while mod>eps and n<nmax  :
        Arg =  [Params , th1n , th2n]
        sol = optimize.root(Newton_eq,q,args=Arg, jac=False)
        dt=sol.x
        q = dt
        th1n+=dt[0]
        th2n+=dt[1]
        mod=np.linalg.norm(dt)
        n+=1
    if n<nmax : success = True
    else : success = False
    return np.array([th1n,th2n]) , success 

