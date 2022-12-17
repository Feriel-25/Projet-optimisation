# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 23:54:31 2022

@author: 
"""
import numpy as np
from scipy import optimize 
from Functions import Newton_eq,F
from Graphics import *

def Newton(TH0,Params,eps,nmax,IsoV=False,Disp=False):
    """
    

    Parameters
    ----------
    TH0 : Configuration Initialle
    
    Params :  Vecteur contenant L1,L2 et un vecteur numPy X du point desiré
    
    eps : La precision 
    
    nmax : Nombre d'iteration Maximale


    Returns
    -------
    [th1n,th2n] : Vecteur numpy de la  Solution a la fin du calcul
    success : True : L'algorithme a converger vers la solution
           False : L'algorithme a depasser le nombre d'iteration avant max avant de converger vers la solution

    """
    mod=1
    th1n=TH0[0]
    th2n=TH0[1]
    n=0
    q = [np.pi/4,np.pi/4]
    if IsoV:
        plt.figure()
    while mod>eps and n<nmax  :
        Arg =  [Params , th1n , th2n]
        # Resolution du systeme matricielle HRnorm(thn)dth+gradRnorm(thn) = 0
        sol = optimize.root(Newton_eq,q,args=Arg, jac=False)
        dt=sol.x
        #Mise a jour de l'estimation initiale
        q = dt
        if IsoV:
            isoValeurs(Params,100,100,25,[th1n,th2n],[th1n+dt[0],th2n+dt[1]]) 
        
        th1n+=dt[0]
        th2n+=dt[1]
        
        # Calcule du module de dt
        mod=np.linalg.norm(dt)
        n+=1
        if Disp :
            print(f" Iterations : {n} \t |dtheta| = {mod}")
    if n<nmax : 
        success = True
        if Disp :
            print(f"Minimum trouve apres {n} iterations")
            print(f"les angles optimaux = {(th1n,th2n)}")
            print(f"Le point (X,Y) = {F(Params[0],Params[1],th1n,th2n)}")
    else : 
        success = False
        print("Erreur Method de newton a echoue ")
    #if IsoV:
        #plt.show()
    return np.array([th1n,th2n]) , success 

#Params = [3,3,[3,0]]
#zawaya,success=Newton([np.pi/4,np.pi/4],Params,0.0000001,100,Disp=True,IsoV=True)

