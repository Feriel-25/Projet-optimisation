# -*- coding: utf-8 -*-
"""


"""
import numpy as np
from scipy import optimize 
from Functions import R , R_norm ,F
from Gradient import  GradienDecent
from Newton import Newton
from Graphics import Interp , animate ,  isoValeurs
import matplotlib.pyplot as plt

L1=3
L2=3
X=np.array([2,1])
Params = [L1,L2,X]
init_config=[np.pi/4,np.pi/4]
   
 
    

 # Methode 1 : Root
print("Methode Root : ")
rt=optimize.root(R, [np.pi/4,np.pi/4],args = Params, jac=False)
#print(rt.success)
if rt.success==True:
    print(f"Les angles optimaux = {rt.x} \nLe point (X,Y) ={F(L1,L2,rt.x[0],rt.x[1])}")
    
else:
    print("Non reachable point")
 # Methode 2 Minimize

min_R_norm=optimize.minimize(R_norm,[np.pi/4,np.pi/4],args = Params)

print("Methode Minimize : ")
print(f"les angles optimaux ={min_R_norm.x} \nLe point (X,Y) ={F(L1,L2,min_R_norm.x[0],min_R_norm.x[1])}")


#Gradient a pas Fixe
aplha_init = 0.01


resultat = GradienDecent(init_config,aplha_init,Params,10e-4,100) 
print("Methode du gradient a pas fixe : ")
print(f"les angles optimaux = {resultat} \nLe point (X,Y) = {F(L1,L2,resultat[0],resultat[1])}")

"""
# Newton 
Params = [L1 , L2 , np.array([4,2])]
res , succ =  Newton ([np.pi/4,np.pi/4],Params,10e-4,1000)
print("Methode de Newton")
print(f"les angles optimaux = {res}")

print(f"Le point (X,Y) = {F(L1,L2,res[0],res[1])}")
"""


Xf=[2,4]
Xi=[6,0]
N=50
x,y=Interp(Xi,Xf,N)
X=[x[0],y[0]]

q = [-2,0] 

fig = plt.figure()
for i in  range(N): 
    X=[x[i],y[i]]
    Params[2] = X
    rt=optimize.root(R,q,args=Params, jac=False)
    rt = GradienDecent(q,aplha_init,Params,10e-4,100) 
    plt.plot(x,y,'r', label = 'droite')
    animate(rt,Params)
    q = rt
    
    
    
    