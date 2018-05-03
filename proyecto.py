#! /bin/env python

import numpy as np
import scipy.linalg as linalg
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.pyplot import ion
from math import *
import sys
from matplotlib.font_manager import FontProperties

def re(y, tobs, h, dobs):
    nobs=np.array(np.floor(tobs/h), dtype=np.int64)
    r = []
    for i in range(0, len(y)):
        r.append([0,0])
    for n in range(0, len(r)):
        if n in nobs:
            pos = np.where(nobs == n)[0][0]
            dobs_pos_0 = dobs[pos][0]
            dobs_pos_1 = dobs[pos][1]
            r[n][0] = dobs_pos_0 - y[n][0]
            r[n][1] = dobs_pos_1 - y[n][1]
    r = np.array(r)
    return r

def gasoilf( t, y, a ):
    """ f fonction définissant le système d'EDO
    t: temps
    y: solution
    a: paramètre 
    yprime: valeur de la fonction
    """
    f1 = -(a[0]+a[2])*(y[0]**2)
    f2 = a[0]*pow(y[0],2) - a[1]*y[1]
    return np.array([f1,f2])
    # A faire par vous même
    # Cf texte projet

def gasoildyf( t, y, a ):
    """ gasoildyf Jacobien de gasoilf par rapport à y
       t: temps
       y: solution
       a: paramètre 
       dyf: jacobienne / y
    """
    df1dy1 = -a[0]-a[2]
    df1dy2 = 0
    df2dy1 = 2*a[0]*y[0]
    df2dy2 = -a[1]
    return np.array([[df1dy1, df1dy2], [df2dy1, df2dy2]])
    # A faire par vous même
    # Cf texte projet

def gasoildaf( t, y, a ):
    """ gasoildyf Jacobien de gasoilf par rapport à y
       t: temps
       y: solution
       a: paramètre 
       daf: jacobienne / a
    """
    df1da1 = -pow(y[0],2)
    df1da2 = 0
    df1da3 = -pow(y[0],2)
    df2da1 = pow(a[0],2)
    df2da2 = -y[1]
    df2da3 = 0
    return np.array([[df1da1, df1da2, df1da3], [df2da1, df2da2,df2da3]])
    # A faire par vous même
    # Cf texte projet

def direct(f, t0, tf, h, a, y):
    """direct  integration EDO 
    
    f: fonction définissant l'EDO
    prototype yprim = f(t,y,a)
    t0, tf: intervalle d'intégration
    h:  pas de discrétisation
    a: paramètres du système
    y: condition initiale

    etat: solution sur tout l'intervalle 
    taille: Nt x length(y) 
    """

    Nt = int((tf-t0)/h)
    etat=[y];
    for i in range(1,Nt+1):
        etat.append([0,0])

    for n in range(1,len(etat)):
        # a completer
        # programmer une méthode d'EUler explicite
        etat[n] = etat[n-1] + h*f(t0, etat[n-1], a)
        #etat=np.vstack((etat, y))
    etat = np.array(etat)
    return etat

def linearise( jyf, t0, tf, h, a, y, b ): #había un dy entre a et y
    """linearise Intégration du système linéarisé de l'EDO

    jyf: jacobienne en y de la fonction définissant l'EDO
    prototype yprim = jyf(t,y,a)
    t0, tf: intervalle d'intégration
    h:  pas de discrétisation
    a: paramètres du système
    y: etat direct, taille (nbre equations) x (nbre pas de temps)
    b: terme source (même taille que y)
    
    detat: solution sur tout l'intervalle 
    taille: même taille que y 
    """
    dy = [[0,0]]
    for n in range(1, len(y)):
        dy.append((b[n-1] - jyf(t0,y[n-1],a).dot(dy[n-1]))*h + dy[n-1])
    dy = np.array(dy)

    # A faire par vous même
    # Cf direct
    detat = dy
    return detat

def adjoint(jyf,  t0, tf, h, a, y, r ):
    """adjoint  Intégration du système adjoint

    jyf: jacobienne en y de la fonction définissant l'EDO
    prototype yprim = jyf(t,y,a)
    t0, tf: intervalle d'intégration
    h:  pas de discrétisation
    a: paramètres du système
    y: etat direct, taille (nbre equations) x (nbre pas de temps)
    r: terme source (même taille que y)
    
    etatdj: solution sur tout l'intervalle 
    taille: même taille que y 
    """
    Nt = int((tf-t0)/h)
    p =[]
    for i in range(0,len(y)):
        p.append([0,0])
    n = len(p)-1
    while n >= 1:
        jyf_t = jyf(t0,y[n],a).transpose()
        jyf_t_par_p = jyf_t.dot(p[n])
        p[n-1] =  h*(r[n] - jyf_t_par_p) + p[n]
        n -= 1
    etadj = np.array(p)
    # A faire par vous même
    # Cf Direct
    # !! Temps rétrograde

    return etadj
        
def tstadj(f, jyf, t0, tf, h, y0, aref,r ):
    """tstadj Test du produit scalaire pour valider l'état adjoint

    f, jyf fonction définissant l'EDO et sa jacobienne
    t0, tf, h, y0: pour l'équation différentielle (cf direct)
    aref: paramètre où on calcule l'adjoint
    
    bp = produit scalaire (b, p)
    dy2 = produit scalaire (dy, dy)
    avec b aléatoire, 
    dy = solution état direct linéarisé (2nd membre b)
    p = solution état adjoint (2nd membre dy)
    """
    yref= direct(f, t0, tf, h, aref, y0);
    b = [] 
    for i in range(0, len(yref)):
        b_fil = np.random.random_sample(2)
        b.append([b_fil[0],b_fil[1]])
    b = np.array(b)
    dy = linearise( jyf, t0, tf, h, aref, yref, b)    
    p = adjoint(jyf,  t0, tf, h, aref, yref, r )
    bp = 0 
    rdy = 0
    for n in range(0, len(p)):
        p_n = p[n]
        b_n = b[n]
        pb = p_n*b_n
        bp += np.sum(pb)
        dy_n = dy[n]
        r_n =r[n]
        dy_n_2 = dy_n*r_n
        rdy += np.sum(dy_n_2)

    # A compléter
    return bp, rdy

def simul( a, f, jyf, jaf, t0, tf, h, y0, tobs, dobs ):
    """ simul Fonction coût pour les moindres ccarés
    
    f, jyf, jaf, fonctions définissant l'EDO
    t0, tf, h, y0: paramètres pour l'EDO (cf direct)
    a: paramètre où on calcule le gradient
    tobs, dobs: instant et valeurs des observations
    
    cout: valeur de la fonction 
    """
    y = direct(f, t0, tf, h, a, y0)
    #nobs=np.array(np.floor(tobs/h), dtype=np.int64)
    nobs=np.array(np.floor(tobs/h), dtype=np.int64)
    r = re(y, tobs, h, dobs)
    cout = 0
    for n in range(0,len(r)):
        cout += 0.5*(pow(r[n][0],2) + pow(r[n][1],2))
        
    return cout
    # Cette fonction doit calculer cout = 1/2 (y(tobs, a) - dobs)^2
    # où y(a) est le résultat de l'état direct
    
    # A faire par vous même

   

def gradient( a, f, jyf, jaf, t0, tf, h, y0, tobs, dobs ):
    """gradient  Calcul du gradient de la fonctionnelle définie dans simul
    
    % f, jyf, jaf, fonctions définissant l'EDO
    % t0, tf, h, y0: paramètres pour l'EDO (cf direct)
    % a: paramètre où on calcule le gradient
    % tobs, dobs: instant et valeurs des observations
    
    % gradient
    """

    y = direct(f, t0, tf, h, a, y0);
    #r=np.zeros(y.shape);
    #nobs=np.array(np.floor(tobs/h), dtype=np.int64)
    nobs=np.array(np.floor(tobs/h), dtype=np.int64)
    r = re(y, tobs, h, dobs)
    p = adjoint(jyf,  t0, tf, h, a, y, r )
    grad = [0.0,0.0,0.0]
    for n in range(0, len(y)):
        grad +=  jaf(t0, y[n], a).transpose().dot(p[n])
    g = grad
    

    # A faire par vous même
    
    return g
        

def tstgrad( f, jyf, jaf, t0, tf, h, y0, a0, delta, tobs, dobs ):
    """ tstgrad validation du gradient par différences finies
    
     f, jyf, jaf, fonctions définissant l'EDO
     t0, tf, h, y0: paramètres pour l'EDO (cf direct)
     a0: paramètre où on calcule le gradient
     (vecteur) des pas utilisés pour les différences finies
     tobs, dobs: instant et valeurs des observations
    
     gref: dérivée directionnelle état adjoint (vecteur (1, 1,...,1))
     g = calculs par DF
    """
    
    cref=simul(a0, f, jyf, jaf, t0, tf, h, y0, tobs, dobs);
    gref=gradient(a0, f, jyf, jaf, t0, tf, h, y0, tobs, dobs);
    gref=np.sum(np.sum(gref));
    
    c=[];
    for i in np.arange(delta.size):
        c=np.hstack((c, simul(a0+delta[i], f, jyf, jaf, t0, tf, h, y0, tobs, dobs)));
        
    g=(c-cref)/delta;
    return gref, g


def main():

    #t0, tf, h, y0 paramètres pour l'éqaution différentielle (cf direct)
    #tobs: instants d'observation
    #Neq, Nparams: nbre d'équations, de paramètres
    #datafile: fichier contenant les observations
    #f, dyf, daf : fonctions définissant le système et ses dérivées

    t0=0; tf=1; h=0.002; y0=[1, 0];
    tobs= np.arange(0,21)*0.05;
    Neqs=2; Nparams=3;
    datafile='projet2.dat';
    f=gasoilf
    jyf=gasoildyf
    jaf=gasoildaf
    nobs=np.array(np.floor(tobs/h), dtype=np.int64)
    t= np.arange(t0, tf+h, h)
    aref=np.ones(Nparams);
    yref= direct(f, t0, tf, h, aref, y0);
    #dobs=np.loadtxt('projet2.dat')
    #dobs = np.random.rand(tobs.size, Neqs);
    dobs = yref[nobs,:]
    dobs_l = [np.loadtxt('projet2.dat'),np.random.rand(tobs.size, Neqs),yref[nobs,:]]
    nobs=np.array(np.floor(tobs/h), dtype=np.int64)
    rref_l = []
    for obs in dobs_l:
        rref_l.append(re(yref, tobs, h, obs))
    plt.figure(1)
    y1 = []
    y2 = []
    for i in range(0,len(yref)):
        y1.append(yref[i][0])
        y2.append(yref[i][1])
    y_1, = plt.plot(t,y1, label='y1')
    y_2, = plt.plot(t,y2, label='y2')
    plt.legend(handles=[y_1, y_2])
    plt.xlabel('t'); plt.ylabel('y')
    plt.savefig("yref.jpg")
    for rreff in rref_l:
        bp, dy2 = tstadj(f, jyf, t0, tf, h, y0, aref,rreff)
        print('Test du produit scalaire : bp={0:18.15e}, rdy={1:18.15e}'.format(bp, dy2))
    
    dobs = np.random.rand(tobs.size, Neqs);
    delta=np.logspace(-2,-11,10) ;  
    [gref, g]=tstgrad(f, jyf, jaf, t0, tf, h, y0, aref, delta, tobs, dobs);
    print("gref: " + str(gref))
    print("g: " + str(g))
    print("(g-gref)/g: " + str((g-gref)/g))

    dobs = yref[nobs,:]
    ainit = aref*1.5;
    resopt = minimize(simul, ainit, method='Nelder-Mead', jac=gradient, args=(f, jyf, jaf, t0, tf, h, y0, tobs, dobs), options={'disp': True})
    aopt=resopt.x
    print("ainit" + str(ainit))
    print("aopt: " + str(aopt))
    ysim=direct(f, t0, tf, h, aopt, y0)
    plt.figure(2)
    plt.plot(t,ysim);
    y1 = []
    y2 = []
    for i in range(0,len(yref)):
        y1.append(ysim[i][0])
        y2.append(ysim[i][1])
    y_1, = plt.plot(t,y1, label='y1')
    y_2, = plt.plot(t,y2, label='y2')
    plt.legend(handles=[y_1, y_2])
    plt.xlabel('t'); plt.ylabel('y')
    plt.savefig("y_test.jpg")
    
    dobs=np.loadtxt('projet2.dat')
    ainit=np.random.rand(Nparams)
    print('a initial {}'.format(ainit))
    yinit= direct(f,t0,tf,h,ainit,y0);
    #resopt = minimize(simul, ainit, method='BFGS', jac=gradient, args=(f, jyf, jaf, t0, tf, h, y0, tobs, dobs), options={'disp': True})
    resopt = minimize(simul, ainit, method='COBYLA', jac=gradient, args=(f, jyf, jaf, t0, tf, h, y0, tobs, dobs), options={'disp': True})
    aopt=resopt.x
    print('a optimal {}'.format(aopt))
    yopt= direct(f,t0,tf,h,aopt,y0);
    plt.figure(3)
    #plt.subplot(223)
    y1_init = []
    y2_init = []
    for i in range(0,len(yinit)):
        y1_init.append(yinit[i][0])
        y2_init.append(yinit[i][1])
    y_1_init, = plt.plot(t,y1_init, label='y1 avec le param. initial')
    y_2_init, = plt.plot(t,y2_init, label='y2 avec le param. initial')

    
    y1_opt = []
    y2_opt = []
    for i in range(0,len(yopt)):
        y1_opt.append(yopt[i][0])
        y2_opt.append(yopt[i][1])
    y_1_opt, = plt.plot(t,y1_opt, label='y1 avec le param optimal')
    y_2_opt, = plt.plot(t,y2_opt, label='y2 avec le param optimal')

    art = []
    dobs1 = []
    dobs2 = []
    for i in range(0,len(dobs)):
        dobs1.append(dobs[i][0])
        dobs2.append(dobs[i][1])
    dobs_1, = plt.plot(tobs, dobs1, "+",label="Observations y1")
    dobs_2, = plt.plot(tobs,dobs2, "x", label="Observations y2")
    plt.xlabel('t'); plt.ylabel('y')
    fontP = FontProperties()
    fontP.set_size('small')
    lgd = plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
    art.append(lgd)
    plt.savefig("yrel.jpg", additional_artists=art, bbox_inches="tight")

if __name__ == "__main__":
    # execute only if run as a script
    main()
    plt.show()