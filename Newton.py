# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 08:59:21 2022
@author: feriel
"""
import numpy as np
from scipy import optimize 
from scipy.interpolate import CubicSpline as CS
import matplotlib.pyplot as plt


th1n = 0
th2n = np.pi/2
TH=np.array([th1n,th2n])
n = 1
L1=1
L2=1
grad=0
Hes=0

X=np.array([0,2])

def F(L1,L2,th1,th2):
    return np.array([L1*np.cos(th1)+L2*np.cos(th1+th2),L1*np.sin(th1)+L2*np.sin(th1+th2)])
'''
def F1(L1,L2,th1,th2,X):
    return np.array([L1*np.cos(th1)+L2*np.cos(th1+th2),L1*np.sin(th1)+L2*np.sin(th1+th2)])-X

print(F(0.1,0.1,np.pi/2,0))
'''



def R(TH):

    return list(np.array([L1*np.cos(TH[0])+L2*np.cos(TH[0]+TH[1]),L1*np.sin(TH[0])+L2*np.sin(TH[0]+TH[1])])-X)



def Rn(TH):
    return np.linalg.norm(np.array(R(TH)))

def Rn2(th1,th2,L1,L2,X):
    return np.power(np.linalg.norm(F(L1,L2,th1,th2,X)),2)
    

    


 

rt=optimize.root(R, [np.pi/4,np.pi/4], jac=False)
#print(rt.success)

# Gradient Pas optimal


def Grad_Re(th1,th2,L1,L2,X):
    x = X[0]
    y = X[1]
    dth1 = 2*x*(L1*np.sin(th1)+L2*np.sin(th1+th2))-2*y*(L1*np.cos(th1)+L2*np.cos(th1+th2))
    dth2 = -2*L1*L2*np.sin(th2)+2*x*L2*np.sin(th1+th2)-2*y*L2*np.sin(th1+th2)
    return np.array([dth1,dth2])


def Hess_Re(th1,th2,L1,L2,X):
    x = X[0]
    y = X[1]
    dth11=   2*x*(L1*np.cos(th1)+L2*np.cos(th1+th2))+2*y*(L1*np.sin(th1)+L2*np.sin(th1+th2))
    dth12 =  2*x*L2*np.cos(th1+th2)+2*y*L2*np.sin(th1+th2)
    dth21=   2*x*L2*np.cos(th1+th2)-2*y*L2*np.cos(th1+th2)
    dth22 = -2*L1*L2*np.cos(th2)+2*x*L2*np.cos(th1+th2)-2*y*L2*np.cos(th1+th2)
    return np.array([[dth11,dth12],[dth21,dth22]])




def Newton_eq(delta_theta):
    #grad=Grad_Re(delta_theta[0],delta_theta[1],L1,L2,X)
    #Hes=Hess_Re(delta_theta[0],delta_theta[1],L1,L2,X)
    '''
    Hes=np.array([[1,0],[0,1]])
    print(np.shape(Hes))
    grad=np.array([1,0])

    '''
   
    return [Hes[0][0]*delta_theta[0]+Hes[0][1]*delta_theta[1]+grad[0],Hes[1][0]*delta_theta[0]+Hes[1][1]*delta_theta[1]+grad[1]]

def Newton(eps,TH,X,nmax):
    mod=1
    th1n=TH[0]
    th2n=TH[1]
    n=0
    while mod>eps and n<nmax  :
        print(grad)
        #grad=Grad_Re(th1n,th2n,L1,L2,X)
        Hes=Hess_Re(th1n,th2n,L1,L2,X)
        sol = optimize.root(Newton_eq,[np.pi/4,np.pi/4], jac=False)
        dt=sol.x
        th1n+=dt[0]
        th2n+=dt[1]
        mod=np.linalg.norm(dt)
        n+=1
    if n<nmax :
        print('youpiiiii')
        print(f'[{th1n},{th2n}]')
        print(F(L1,L2,th1n,th2n))

        

#print(Newton(0.1,TH,X,1000))

mod=1
th1n=TH[0]
th2n=TH[1]
n=0
eps=0.1
nmax=1000
while mod>eps and n<nmax  :
    
    grad=Grad_Re(th1n,th2n,L1,L2,X)
    Hes=Hess_Re(th1n,th2n,L1,L2,X)
    sol = optimize.root(Newton_eq,[0,0], jac=False)
    dt=sol.x
    th1n+=dt[0]
    th2n+=dt[1]
    mod=np.linalg.norm(dt)
    n+=1
    print(f'[{th1n%(2*np.pi)},{th2n%(2*np.pi)}]')
if n<nmax :
    print('youpiiiii')
    print(f'[{th1n%(2*np.pi)},{th2n%(2*np.pi)}]')
    print(F(L1,L2,th1n,th2n))