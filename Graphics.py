# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 23:48:57 2022

@author: khaled
"""
import numpy as np
from scipy.interpolate import CubicSpline as CS
import matplotlib.pyplot as plt
from Functions import R_norm
    
def Interp(Xi,Xf,N,traj=None,Xm=None):
    """
    

    Parameters
    ----------
    Xi : Point initialle 
    Xf : Point finale 
    N : Nombre de points
    traj : Defini si la trajectoire est libaire ou cubique
        Valeur par default c'est None dont trajectoire lineaire.
    Xm : Point Intermediaire
        Valeur par default c'est None dont trajectoire lineaire.

    Returns
    -------
    Vecteur de points X et Y

    """
    
    x=np.linspace(Xi[0],Xf[0],N)
    if traj==None:
    
        y=np.linspace(Xi[1],Xf[1],N)
        return [x,y]
    
    elif traj=="Cube":
        f=CS([Xi[0],Xm[0],Xf[0]],[Xi[1],Xm[1],Xf[1]])
        return [x,f(x)]
    
    


def animate (TH,Params):
    """
    

    Parameters
    ----------
    TH : Les angles articulaire du robot TH = [theta1,theta2]
    Params : Vecteur contenant L1,L2 et un vecteur numPy X du point desiré

    Returns
    -------
    None.

    """
    L1 = Params[0]
    L2 = Params[1]
    plt.xlim([-4,8])
    plt.ylim([-4,8])
    x1, y1 = [0, L1*np.cos(TH[0])], [0, L1*np.sin(TH[0])]
    x2, y2 = [L1*np.cos(TH[0]),L1*np.cos(TH[0])+L2*np.cos(TH[0]+TH[1])], [L1*np.sin(TH[0]),L1*np.sin(TH[0])+ L2*np.sin(TH[0]+TH[1])]
    plt.plot(x1, y1,x2,y2 ,marker = 'o',color='b',linewidth=10)
    plt.scatter(x1[1],y1[1], s=100,marker='o',color='k',linewidths=20)
    plt.scatter(x2[1],y2[1], s=100,marker='o',color='k',linewidths=20)
    
    


def isoValeurs (Params,nx,ny, nIso,thn=[None,None],thn_1=[None,None]) : 
    """

    Parameters
    ----------
    Params : Vecteur contenant L1,L2 et un vecteur numPy X
    nx : Nombre de points en x
        
    ny :   Nombre de points en y.
    nIso :Nombre d'Isovaleurs a dessiner
    
    thn : Vecteur contenant les angles articulaires a l'instant  n , optional
       Valeur par default c'est [None,None].
    thn_1 : Vecteur contenant les angles articulaires a l'instant  n+1 , optional
       Valeur par default c'est [None,None].

    Returns
    -------
    None.

    """
    # Définition du domaine de tracé
    xmin, xmax, nx = -np.pi, np.pi, nx
    ymin, ymax, ny = -np.pi, np.pi, ny
    # Discrétisation du domaine de tracé
    x1d = np.linspace(xmin,xmax,nx)
    y1d = np.linspace(ymin,ymax,ny)
    x2d, y2d = np.meshgrid(x1d, y1d)
  
    #Tracé des isovaleur 
    plt.contour(x2d,y2d,R_norm([x2d,y2d],Params),nIso)
    # Si thn et thn_1 sont pas egale a [None,None] on trace les isovaleurs et les angles (les points)
    if (all(thn)!=None and all(thn_1)!=None) : 
        plt.scatter(thn[0],thn[1],color='black')
        plt.plot([thn[0], thn_1[0]],[thn[1], thn_1[1]],"--r")
    plt.title('Isovaleurs')
    plt.xlabel('Valeurs de theta1')
    plt.ylabel('Valeurs de theta2')
    plt.grid()
    