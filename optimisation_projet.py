# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 08:59:21 2022

@author: feriel
"""
import numpy as np
from scipy import optimize 
from scipy.interpolate import CubicSpline as CS
import matplotlib.pyplot as plt
def F(L1,L2,th1,th2):
    return np.array([L1*np.cos(th1)+L2*np.cos(th1+th2),L1*np.sin(th1)+L2*np.sin(th1+th2)])

def F1(L1,L2,th1,th2,X):
    return np.array([L1*np.cos(th1)+L2*np.cos(th1+th2),L1*np.sin(th1)+L2*np.sin(th1+th2)])-X

print(F(0.1,0.1,np.pi/2,0))



def R(TH):

    return list(np.array([L1*np.cos(TH[0])+L2*np.cos(TH[0]+TH[1]),L1*np.sin(TH[0])+L2*np.sin(TH[0]+TH[1])])-X)



def Rn(TH):
    return np.linalg.norm(np.array(R(TH)))

def Rn2(th1,th2,L1,L2,X):
    return np.power(np.linalg.norm(F1(L1,L2,th1,th2,X)),2)
    
    
    
def Interp(Xi,Xf,N,traj=None,Xm=None):
    
    x=np.linspace(Xi[0],Xf[0],N)
    if traj==None:
    
        y=np.linspace(Xi[1],Xf[1],N)
        return [x,y]
    
    elif traj=="Cube":
        f=CS([Xi[0],Xm[0],Xf[0]],[Xi[1],Xm[1],Xf[1]])
        return [x,f(x)]
    

 
L1=0.1
L2=0.1

X=np.array([0.15,0.04])
rt=optimize.root(R, [np.pi/4,np.pi/4], jac=False)
#print(rt.success)
if rt.success==True:
    print(rt.x[0],rt.x[1])
    
    print(F(L1,L2,rt.x[0],rt.x[1]))
else:
    print(F(L1,L2,rt.x[0],rt.x[1]))
    print("Non reachable point")
    
znijo=optimize.minimize(Rn,[np.pi/4,np.pi/4])

print(znijo.x)
print(F(L1,L2,znijo.x[0],znijo.x[1]))
#fig=plt.figure()

#plt.plot(Interp(Xi=[0.2,0],Xf=[0,0.2],N=100)[0], Interp(Xi=[0.2,0],Xf=[0,0.2],N=100)[1],'r', label = 'droite')


x,y= Interp(Xi=[0,0.2],Xf=[0.2,0],N=100,traj="Cube",Xm=[0.05,0.1])
plt.plot(x,y,'r', label = 'droite')


init_ang=[np.pi/4,np.pi/4]
"""for i in range(len(x)):
    X=np.array([x[i],y[i]])
    znijo=optimize.minimize(Rn,init_ang)
    init_ang=znijo.x
    #print(znijo.x[0],znijo.x[1])
    res=F(L1,L2,znijo.x[0],znijo.x[1])
    print(res)
    #fig.plot(res[0], res[1],'k')
    plt.scatter(res[0], res[1],color="black")
    
    #plt.plot(res[0], res[1],'k')
"""    

# Gradient Pas optimal


def Grad_Re(th1,th2,L1,L2,X):
    x = X[0]
    y = X[1]
    dth1 = 2*x*(L1*np.sin(th1)+L2*np.sin(th1+th2))-2*y*(L1*np.cos(th1)+L2*np.cos(th1+th2))
    dth2 = -2*L1*L2*np.sin(th2)+2*x*L2*np.sin(th1+th2)-2*y*L2*np.sin(th1+th2)
    return np.array([dth1,dth2])

th1n = 0
th2n = 0
n = 1
def Sys_Eqt0(x):
    return [
        Rn2((th1n-x[0]*Grad_Re(th1n,th2n,L1,L2,X)[0]),(th2n-x[0]*Grad_Re(th1n,th2n,L1,L2,X)[1]),L1,L2,X)
           ]
while (True) :
    res = optimize.root(Sys_Eqt0,[0.1], jac=False) 
    alpha = res.x[0]
    th1n_1 = th1n - alpha*Grad_Re(th1n,th2n,L1,L2,X)[0]
    th2n_1 = th2n - alpha*Grad_Re(th1n,th2n,L1,L2,X)[1]
    if (Rn2(th1n_1,th2n_1,L1,L2,X)>Rn2(th1n,th2n,L1,L2,X)):
        print("Minimum trouve")
        break
    th1n,x2n = th1n_1,th2n_1
    print(n)
    if(n>10000) :  break 
    n+= 1 
print( th1n_1,th2n_1)
print(f"Le nombre d'iterations pour cette methode est : {n} iterations")    

print(F(L1,L2, th1n_1,th2n_1))

