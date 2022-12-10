# -*- coding: utf-8 -*-
"""

"""
import numpy as np

def F(L1,L2,th1,th2):
    return np.array([L1*np.cos(th1)+L2*np.cos(th1+th2),L1*np.sin(th1)+L2*np.sin(th1+th2)])




def R(TH,Params):
    L1= Params[0]
    L2 = Params[1]
    X = Params[2]
    return np.array([L1*np.cos(TH[0])+L2*np.cos(TH[0]+TH[1])-X[0],L1*np.sin(TH[0])+L2*np.sin(TH[0]+TH[1])-X[1]])



def R_norm(TH,Params):
    L1= Params[0]
    L2 = Params[1]
    X = Params[2]
    return (L1*np.cos(TH[0])+L2*np.cos(TH[0]+TH[1])-X[0])**2 + (L1*np.sin(TH[0])+L2*np.sin(TH[0]+TH[1])-X[1])**2



    
def Grad_Re(th1,th2,Params):
    L1= Params[0]
    L2 = Params[1]
    X = Params[2]
    x = X[0]
    y = X[1]
    dth1 = 2*x*(L1*np.sin(th1)+L2*np.sin(th1+th2))-2*y*(L1*np.cos(th1)+L2*np.cos(th1+th2))
    dth2 = -2*L1*L2*np.sin(th2)+2*x*L2*np.sin(th1+th2)-2*y*L2*np.cos(th1+th2)
    return np.array([dth1,dth2])


def Hess_Re(th1,th2,Params):
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
    Params = Arg [0]
    th1n = Arg[1]
    th2n = Arg [2]
    grad=Grad_Re(th1n,th2n,Params)
    Hes=Hess_Re(th1n,th2n,Params)

    return [Hes[0][0]*delta_theta[0]+Hes[0][1]*delta_theta[1]+grad[0],Hes[1][0]*delta_theta[0]+Hes[1][1]*delta_theta[1]+grad[1]]


