# -*- coding: utf-8 -*-
"""

"""
import numpy as np


def F(L1,L2,th1,th2):
    """
    

    Parameters
    ----------
    L1 : Longeur du premier segment du robot
    L2 : Longeur du deuxieme segment du robot
    th1 : L'angle articulaire theta 1 
    th2 : L'angle articulaire theta 1 

    Returns
    -------
    Un vecteur numpy de la position de l'organe terminale du Robot 

    """
    return np.array([L1*np.cos(th1)+L2*np.cos(th1+th2),L1*np.sin(th1)+L2*np.sin(th1+th2)])




def R(TH,Params):
    """
    

    Parameters
    ----------
    TH : Les angles articulaire du robot TH = [theta1,theta2]
    Params : Vecteur contenant L1,L2 et un vecteur numPy X du point desiré

    Returns
    -------
    La fonction residu a un point defini

    """
    L1= Params[0]
    L2 = Params[1]
    X = Params[2]
    return np.array([L1*np.cos(TH[0])+L2*np.cos(TH[0]+TH[1])-X[0],L1*np.sin(TH[0])+L2*np.sin(TH[0]+TH[1])-X[1]])



def R_norm(TH,Params):
    """
    

    Parameters
    ----------
    TH : Les angles articulaire du robot TH = [theta1,theta2]
    Params : Vecteur contenant L1,L2 et un vecteur numPy X du point desiré
    
    Returns
    -------
    La norm carré de la fonction residu a un un point defini

    """
    L1= Params[0]
    L2 = Params[1]
    X = Params[2]
    return (L1*np.cos(TH[0])+L2*np.cos(TH[0]+TH[1])-X[0])**2 + (L1*np.sin(TH[0])+L2*np.sin(TH[0]+TH[1])-X[1])**2



    
def Grad_Re(th1,th2,Params): 
    """
    Grad_Re calcule le gradient de la norm carré de la fonction residu

    Parameters
    ----------
    th1 : L'angle articulaire theta 1 
    th2 : L'angle articulaire theta 1 
   
    Params : Vecteur contenant L1,L2 et un vecteur numPy X du point desiré

    Returns
    -------
    Gradient de R_norm

    """
    L1= Params[0]
    L2 = Params[1]
    X = Params[2]
    x = X[0]
    y = X[1]
    dth1 = 2*x*(L1*np.sin(th1)+L2*np.sin(th1+th2))-2*y*(L1*np.cos(th1)+L2*np.cos(th1+th2))
    dth2 = -2*L1*L2*np.sin(th2)+2*x*L2*np.sin(th1+th2)-2*y*L2*np.cos(th1+th2)
    return np.array([dth1,dth2])


def Hess_Re(th1,th2,Params):
    """
    Hess_Re calcule La matrice Hessiene de la la norm carré de la fonction residu

    Parameters
    ----------
    th1 : L'angle articulaire theta 1 
    th2 : L'angle articulaire theta 1 
   
    Params : Vecteur contenant L1,L2 et un vecteur numPy X du point desiré

    Returns
    -------
    Matrice Hessienne de R_norm

    """
    L1 = Params[0]
    L2 = Params[1]
    X = Params[2]
    x = X[0]
    y = X[1]
    dth11=   2*x*(L1*np.cos(th1)+L2*np.cos(th1+th2))+2*y*(L1*np.sin(th1)+L2*np.sin(th1+th2))
    dth12 =  2*x*L2*np.cos(th1+th2)+2*y*L2*np.sin(th1+th2)
    dth21=   2*x*L2*np.cos(th1+th2)+2*y*L2*np.sin(th1+th2)
    dth22 = -2*L1*L2*np.cos(th2)+2*x*L2*np.cos(th1+th2)+2*y*L2*np.sin(th1+th2)
    return np.array([[dth11,dth12],[dth21,dth22]])

def Newton_eq(delta_theta,Arg):
    """
    Newton_eq est utiliser pour resoudre l'equation HRnorm(thn)dth+gradRnorm(thn) = 0'

    Parameters
    ----------
    delta_theta : Solution du systeme matricielle HRnorm(thn)dth+gradRnorm(thn) = 0
    
    Arg : Vecteur contenant thn[1] et thn[2], ainsi que le vecteur Params
    
    Returns
    -------
    Une definition du systeme d'equation

    """
    Params = Arg [0]
    th1n = Arg[1]
    th2n = Arg [2]
    grad=Grad_Re(th1n,th2n,Params)
    Hes=Hess_Re(th1n,th2n,Params)

    return [Hes[0][0]*delta_theta[0]+Hes[0][1]*delta_theta[1]+grad[0],Hes[1][0]*delta_theta[0]+Hes[1][1]*delta_theta[1]+grad[1]]


