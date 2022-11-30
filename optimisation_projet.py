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
X=np.array([4,3])

def F(L1,L2,th1,th2):
    return np.array([L1*np.cos(th1)+L2*np.cos(th1+th2),L1*np.sin(th1)+L2*np.sin(th1+th2)])

def F1(L1,L2,th1,th2,X):
    return np.array([L1*np.cos(th1)+L2*np.cos(th1+th2),L1*np.sin(th1)+L2*np.sin(th1+th2)])-X


def R(TH):
    return np.array([L1*np.cos(TH[0])+L2*np.cos(TH[0]+TH[1])-X[0],L1*np.sin(TH[0])+L2*np.sin(TH[0]+TH[1])]-X[1])



def R_norm(TH):
    return (L1*np.cos(TH[0])+L2*np.cos(TH[0]+TH[1])-X[0])**2 + (L1*np.sin(TH[0])+L2*np.sin(TH[0]+TH[1])-X[1])**2

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
    

 # Methode 1 : Root
print("Methode Root")
rt=optimize.root(R, [np.pi/4,np.pi/4], jac=False)
#print(rt.success)
if rt.success==True:
    print(f"Les angles optimaux :{rt.x[0]},{rt.x[1]}")
    
    print(f"Position de l'organe terminale du robot avec les angles optimaux: {F(L1,L2,rt.x[0],rt.x[1])}")
else:
    print(f"Position de l'organe terminale du robot : {F(L1,L2,rt.x[0],rt.x[1])}")
    print("Non reachable point")
 # Methode 2 Minimize
print("Methode Minimize")
min_R_norm=optimize.minimize(R_norm,[np.pi/4,np.pi/4])

print(f"Les angles optimaux :{min_R_norm.x}")
print(f"Position de l'organe terminale du robot avec les angles optimaux: {F(L1,L2,min_R_norm.x[0],min_R_norm.x[1])}")

#fig=plt.figure()
#plt.plot(Interp(Xi=[0.2,0],Xf=[0,0.2],N=100)[0], Interp(Xi=[0.2,0],Xf=[0,0.2],N=100)[1],'r', label = 'droite')


x,y= Interp(Xi=[0,0.2],Xf=[0.2,0],N=100,traj="Cube",Xm=[0.05,0.1])
plt.plot(x,y,'r', label = 'droite')


init_config=[np.pi/4,np.pi/4]
"""for i in range(len(x)):
    X=np.array([x[i],y[i]])
    min_R_norm=optimize.minimize(Rn,init_config)
    init_config=min_R_norm.x
    #print(min_R_norm.x[0],min_R_norm.x[1])
    res=F(L1,L2,min_R_norm.x[0],min_R_norm.x[1])
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
    if(n>10000) :  break 
    n+= 1 
print( th1n_1,th2n_1)
print(f"Le nombre d'iterations pour cette methode est : {n} iterations")    

print(F(L1,L2, th1n_1,th2n_1))

#plotting

def animate (TH,fig):  
    
    L1=3
    L2=3
    x1, y1 = [0, L1*np.cos(TH[0])], [0, L1*np.sin(TH[0])]
    x2, y2 = [L1*np.cos(TH[0]),L1*np.cos(TH[0])+L2*np.cos(TH[0]+TH[1])], [L1*np.sin(TH[0]),L1*np.sin(TH[0])+ L2*np.sin(TH[0]+TH[1])]
 
    
    plt.plot(x1, y1,x2,y2 ,marker = 'o',color='b',linewidth=10)
    plt.scatter(x1[1],y1[1], s=100,marker='o',color='k',linewidths=20)
    plt.scatter(x2[1],y2[1], s=100,marker='o',color='k',linewidths=20)
    plt.show()


#Gradient a pas Fixe
aplha_init = 0.3

def Grad_Re(th1,th2,L1,L2,X):
    x = X[0]
    y = X[1]
    dth1 = 2*x*(L1*np.sin(th1)+L2*np.sin(th1+th2))-2*y*(L1*np.cos(th1)+L2*np.cos(th1+th2))
    dth2 = -2*L1*L2*np.sin(th2)+2*x*L2*np.sin(th1+th2)-2*y*L2*np.sin(th1+th2)
    return np.array([dth1,dth2])
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
        thn = thn_1
        n=n+1
    return thn
TH00 = np.array([0,0])
plt.figure(figsize=(6,6))
resultat = GradienDecent(TH00,alpha,L1,L2,10e-4,100,X) 

print("methode du gradient")
print(f"les angles optimaux : { resultat}")
print(f"Position de l'organe terminale du robot avec les angles optimaux: {F(L1,L2,resultat[0],resultat[1])}")

# Définition du domaine de tracé
xmin, xmax, nx = -np.pi, np.pi, 100
ymin, ymax, ny = -np.pi, np.pi, 100
# Discrétisation du domaine de tracé
x1d = np.linspace(xmin,xmax,nx)
y1d = np.linspace(ymin,ymax,ny)
x2d, y2d = np.meshgrid(x1d, y1d)
print(np.shape(x2d),np.shape(y2d))
# Tracé des isovaleurs de f1
nIso = 100
#Tracé des isovaleur pour une seule iteration 
plt.figure("fig")

plt.contour(x2d,y2d,R_norm([x2d,y2d]),nIso)
plt.title('Isovaleurs')
plt.xlabel('Valeurs de x1')
plt.ylabel('Valeurs de x2')
plt.grid()

Xi=[5,1]
Xf=[1,3]
N=50
x,y=Interp(Xi,Xf,N)

plt.plot(x,y,'r', label = 'droite')
for i in  range(N): 
    X=[x[i],y[i]]
    #angles=GradienDecent(TH00,alpha,L1,L2,10e-4,100,[x[i],y[i]]) 
    rt=optimize.root(R, [np.pi/4,np.pi/4], jac=False)
    print(rt.x)
    animate(rt.x,plt.figure())