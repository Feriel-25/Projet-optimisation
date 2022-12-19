# -*- coding: utf-8 -*-
"""


"""
import numpy as np
from scipy import optimize 
from Functions import R , R_norm , F
from Gradient import  *
from Newton import Newton
from Graphics import  isoValeurs
import matplotlib.pyplot as plt

def Interface(L1,L2):
    print("Veuillez rentrer les coordonnées d'un point desirée : ")
    x = int(input("X :  "))
    y = int(input("Y :  "))
    Params = [L1,L2,np.array([x,y])]
    iso = input("Voulez vous vous voir un tracé des isovaleurs : [o/n]  :")
    if iso == 'o' :
        IsoV = True
    else : IsoV = False 

    print ("Veuillez rentrer la configuration intialle du robot")
    th1 = float(input("Theta1 : "))
    th2 = float(input("Theta2 : "))
    ConfigInit = [th1,th2]
    print("Selectionner une methode de resolution")
    while (True) :
        m = input("1 : Scipy.Optimize  \n2 : Scipy.Minimize \n3 : Gradient \n4 : Newton\n ")
        if (m=='1') :
            resultat = optimize.root(R, [np.pi/4,np.pi/4],args = Params, jac=False)
            print(f"les angles optimaux = {resultat.x} \nLe point X = {F(L1,L2,resultat.x[0],resultat.x[1])}")
            break
        elif (m=='2') :
            resultat = optimize.minimize(R_norm,[np.pi/4,np.pi/4],args = Params)
            print(f"les angles optimaux = {resultat.x} \nLe point X = {F(L1,L2,resultat.x[0],resultat.x[1])}")
            break 

        elif (m == '3'):
            aplhaInit = float(input("Veuillez entrer le pas du gradient alpha : ..."))
            n_max = int(input("Veuillez entrer la valeur maximale d'iteration : ..."))
            eps = float(input("Veuillez entrer la valeur de la precision epsilon : ..."))
            resultat , converge = GradienDecentAmeliore(ConfigInit,aplhaInit,Params,eps,n_max,IsoV=IsoV ,Disp=False)
            print(f"les angles optimaux = {resultat} \nLe point X = {F(L1,L2,resultat[0],resultat[1])}")
            break 
        elif (m =='4'):
            eps = float(input("Veuillez entrer la valeur de la precision epsilon : ..."))
            nmax = int(input("Veuillez entrer la valeur maximale d'iteration : ..."))
            resultat,success=Newton(ConfigInit,Params,eps,nmax,Disp=False,IsoV=IsoV)
            break 

        else :
            print("Veuillez saisir un nombre entre 1 et 4")

