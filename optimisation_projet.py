# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 08:59:21 2022

@author: feriel
"""
import numpy as np
from scipy import optimize 
from scipy.interpolate import CubicSpline as CS
import matplotlib.pyplot as plt
L1=3
L2=3
X=np.array([2,1])

def F(L1,L2,th1,th2):
    return np.array([L1*np.cos(th1)+L2*np.cos(th1+th2),L1*np.sin(th1)+L2*np.sin(th1+th2)])




def R(TH):
    return np.array([L1*np.cos(TH[0])+L2*np.cos(TH[0]+TH[1])-X[0],L1*np.sin(TH[0])+L2*np.sin(TH[0]+TH[1])-X[1]])



def R_norm(TH):
    return (L1*np.cos(TH[0])+L2*np.cos(TH[0]+TH[1])-X[0])**2 + (L1*np.sin(TH[0])+L2*np.sin(TH[0]+TH[1])-X[1])**2


   
    
    
def Interp(Xi,Xf,N,traj=None,Xm=None):
    
    x=np.linspace(Xi[0],Xf[0],N)
    if traj==None:
    
        y=np.linspace(Xi[1],Xf[1],N)
        return [x,y]
    
    elif traj=="Cube":
        f=CS([Xi[0],Xm[0],Xf[0]],[Xi[1],Xm[1],Xf[1]])
        return [x,f(x)]
    

 # Methode 1 : Root
print("Methode Root : ")
rt=optimize.root(R, [np.pi/4,np.pi/4], jac=False)
#print(rt.success)
if rt.success==True:
    print(f"Les angles optimaux = {rt.x} \nLe point (X,Y) ={F(L1,L2,rt.x[0],rt.x[1])}")
    
else:
    print("Non reachable point")
 # Methode 2 Minimize

min_R_norm=optimize.minimize(R_norm,[np.pi/4,np.pi/4])

print("Methode Minimize : ")
print(f"les angles optimaux ={min_R_norm.x} \nLe point (X,Y) ={F(L1,L2,min_R_norm.x[0],min_R_norm.x[1])}")




init_config=[np.pi/4,np.pi/4]


def animate (TH): 
    plt.xlim([-2,8])
    plt.ylim([-2,8])
    x1, y1 = [0, L1*np.cos(TH[0])], [0, L1*np.sin(TH[0])]
    x2, y2 = [L1*np.cos(TH[0]),L1*np.cos(TH[0])+L2*np.cos(TH[0]+TH[1])], [L1*np.sin(TH[0]),L1*np.sin(TH[0])+ L2*np.sin(TH[0]+TH[1])]
 
    
    plt.plot(x1, y1,x2,y2 ,marker = 'o',color='b',linewidth=10)
    plt.scatter(x1[1],y1[1], s=100,marker='o',color='k',linewidths=20)
    plt.scatter(x2[1],y2[1], s=100,marker='o',color='k',linewidths=20)
    plt.show()
def Grad_Re(th1,th2,L1,L2,X):
    x = X[0]
    y = X[1]
    dth1 = 2*x*(L1*np.sin(th1)+L2*np.sin(th1+th2))-2*y*(L1*np.cos(th1)+L2*np.cos(th1+th2))
    dth2 = -2*L1*L2*np.sin(th2)+2*x*L2*np.sin(th1+th2)-2*y*L2*np.cos(th1+th2)
    return np.array([dth1,dth2])


def Hess_Re(th1,th2,L1,L2,X):
    x = X[0]
    y = X[1]
    dth11=   2*x*(L1*np.cos(th1)+L2*np.cos(th1+th2))+2*y*(L1*np.sin(th1)+L2*np.sin(th1+th2))
    dth12 =  2*x*L2*np.cos(th1+th2)+2*y*L2*np.sin(th1+th2)
    dth21=   2*x*L2*np.cos(th1+th2)+2*y*L2*np.sin(th1+th2)
    dth22 = -2*L1*L2*np.cos(th2)+2*x*L2*np.cos(th1+th2)+2*y*L2*np.sin(th1+th2)
    return np.array([[dth11,dth12],[dth21,dth22]])



def isoValeurs (nx,ny, nIso,thn=[None,None],thn_1=[None,None]) : 
    # Définition du domaine de tracé
    xmin, xmax, nx = -np.pi, np.pi, 100
    ymin, ymax, ny = -np.pi, np.pi, 100
    # Discrétisation du domaine de tracé
    x1d = np.linspace(xmin,xmax,nx)
    y1d = np.linspace(ymin,ymax,ny)
    x2d, y2d = np.meshgrid(x1d, y1d)
  
    #Tracé des isovaleur pour une seule iteration 
    plt.figure()
    
    plt.contour(x2d,y2d,R_norm([x2d,y2d]),nIso)
    if (all(thn)!=None and all(thn_1)!=None) : 
        plt.scatter(thn[0],thn[1],color='black')
        plt.plot([thn[0], thn_1[0]],[thn[1], thn_1[1]],"--r")
    plt.title('Isovaleurs')
    plt.xlabel('Valeurs de x1')
    plt.ylabel('Valeurs de x2')
    plt.grid()
    
    
#Gradient a pas Fixe
aplha_init = 0.05

def GradienDecent (TH0,alpha,L1,L2,eps,n_max,x):
    Un = 1
    divide = False
    thn = TH0 
    n=0
    while(np.linalg.norm(Un)>eps and n<n_max ) :
        thn_1 =  thn - alpha * Grad_Re(thn[0],thn[1],L1,L2,x)
        if (R_norm(thn)<R_norm(thn_1)) and not divide  : 
            alpha = aplha_init / 2 
            divide = True
        elif(R_norm(thn)>R_norm(thn_1) and divide):
            alpha = aplha_init
            divide = False
        Un = thn_1 - thn 
        #isoValeurs (100,100,25,thn,thn_1)
        thn = thn_1
        n=n+1
    return thn

resultat = GradienDecent(init_config,aplha_init,L1,L2,10e-4,100,X) 
print("Methode du gradient a pas fixe : ")
print(f"les angles optimaux = {resultat} \nLe point (X,Y) = {F(L1,L2,resultat[0],resultat[1])}")


grad = None 
Hes = None

def Newton_eq(delta_theta):
    return [Hes[0][0]*delta_theta[0]+Hes[0][1]*delta_theta[1]+grad[0],Hes[1][0]*delta_theta[0]+Hes[1][1]*delta_theta[1]+grad[1]]

def Newton(eps,TH,X,nmax):
    global Hes 
    global grad
    mod=1
    th1n=TH[0]
    th2n=TH[1]
    n=0
    q = [np.pi/4,np.pi/4]
    while mod>eps and n<nmax  :
        grad=Grad_Re(th1n,th2n,L1,L2,X)
        Hes=Hess_Re(th1n,th2n,L1,L2,X)
        sol = optimize.root(Newton_eq,q, jac=False)
        dt=sol.x
        q = dt
        th1n+=dt[0]
        th2n+=dt[1]
        mod=np.linalg.norm(dt)
        n+=1
    if n<nmax : success = True
    else : success = False
    return np.array([th1n,th2n])

TH=np.array([0.75,0.75])

X=np.array([4,2.2])
#rv = Newton(0.1,TH,X,1000)
print(X)
rv = GradienDecent(TH,aplha_init,L1,L2,10e-4,100,X) 
print(rv)
print(F(L1,L2,rv[0],rv[1]))

#animate(rv)
#isoValeurs (100,100,25)
#plt.scatter(rv[0],rv[1],color='black')

Xf=[2,4]
Xi=[2,6]
N=50
x,y=Interp(Xi,Xf,N)
X=[x[0],y[0]]
isoValeurs (100,100,100)
q = [-0.5,1]

fig = plt.figure()
#q = [1.5,0]
qg = [1.5,0]
for i in  range(N): 
    X=[x[i],y[i]]
    #♠angles=Newton(0.001,q,X,1000)
    rt=optimize.root(R, q, jac=False)
    plt.plot(x,y,'r', label = 'droite')
    animate(rt.x)
    q = rt.x
    #print(F(L1,L2,angles[0],angles[1]))
    
    
    