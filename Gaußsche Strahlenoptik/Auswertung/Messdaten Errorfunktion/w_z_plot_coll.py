# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 23:08:36 2022

@author: Jan-Philipp
"""

import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit 
import matplotlib
from itertools import product, combinations
from scipy import odr

matplotlib.style.use('JaPh') 
plt.ioff()

def Regr(z,w0,z0,M):
    lamb = .0006328
    zR = np.pi*w0**2/lamb*1/M**2
    return w0*np.sqrt(1+(z-z0)**2/zR**2)

def Regr_sp(tup,z):
    return Regr(z,*tup)
    
def plot(CSVNAME):
    PATH = CSVNAME + '.csv'


    p0 = (.1,100,1)

    X, Y, Yerr = np.loadtxt(PATH,delimiter=',')    
      
    
    Xerr = np.ones(len(X))*1
   
    xlabel = r'$\frac{z}{\mathrm{mm}}$' 
    ylabel = r'$\frac{w(z)}{\mathrm{mm}}$'
    
    xlim = (-100,400)
    ylim = (min(Y)*0.9-.1,max(Y)*1.1)

    popt,pcov = curve_fit(Regr,X,Y,p0=p0,maxfev=100000)#,sigma=Yerr,absolute_sigma=True)
    stdDev=np.sqrt(np.diag(pcov))  
    

    #----------------------------------------------------------------------
    FuncModel = odr.Model(Regr_sp)
    data = odr.RealData(X,Y,sx=Xerr,sy=Yerr)
    
    sodr = odr.ODR(data, FuncModel, beta0=list(p0),maxit=100000) # beta0 contains initial parameter values
    regr = sodr.run() #fitting a linear curve, popt contains optimal values for the parameters a and b, pcov contains covariance of popt
    
    popt = regr.beta
    pcov = regr.cov_beta 
    stdDev=np.sqrt(np.diag(pcov))  
    #----------------------------------------------------------------------


    t1 = np.linspace(xlim[0],xlim[1], 10**4)

    fig,ax=plt.subplots(1,1,figsize=(10,10/np.sqrt(2))) 
  
    ax.plot(t1,Regr(t1,*popt), marker = 'None', linestyle = '-',label=r'$w(z)=w_0\cdot\sqrt{1+\frac{(z-z_0)^2}{z_R(w_0,\lambda,M)^2}}$') 
    ax.errorbar(X,Y,yerr=Yerr,xerr=Xerr,marker='x',linestyle='None',label='Messwerte')
    
    for comb1, comb2 in combinations(product([-1,1],repeat=len(stdDev)),2):
        p1 = popt+stdDev*comb1
        p2 = popt+stdDev*comb2
        if np.abs(np.sum(comb1))==len(stdDev) and np.sum(comb2)==-np.sum(comb1):
            ax.fill_between(t1,Regr(t1,*p1),Regr(t1,*p2), alpha = .6, label = r'$1\sigma-\text{Fehlerstreifen}$')
        else: 
            ax.fill_between(t1,Regr(t1,*p1),Regr(t1,*p2), alpha = .6)
    
    
    sigfig = 4
    stdDev_rounded = ['{:g}'.format(float('{:.{p}g}'.format(stdDev[i], p=sigfig))) for i in range(0,len(popt))]
    decimals = [len(str(stdDev_rounded[i].split('.')[-1])) for i in range(0,len(popt))]
    cell_text = [[str(round(popt[i],decimals[i])),str(round(stdDev[i],decimals[i]))] for i in range(0,len(popt))]
    
    the_table = the_table = plt.table(cellText=cell_text,
                      rowLabels=[r'$w_0[\mathrm{mm}]$',r'$z_0[\mathrm{mm}]$',r'$M$'],
                      colLabels=['Wert','Unsicherheit'],
                      loc='bottom',
                      cellLoc='center',
                      bbox=[0.25,-0.3,0.5,0.2],
                      edges='closed')

    ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlabel(xlabel,size=16)
    ax.set_ylabel(ylabel,size=16)
    ax.set_xlim(xlim[0],xlim[1])
    ax.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)

    ax.legend(loc='best',fontsize=15)
    fig.tight_layout()

    #----Save Figure-----------
    plt.savefig(CSVNAME+".pdf",dpi=1200)
    plt.savefig(CSVNAME+".png",dpi=1200)



if __name__ == "__main__":    
   for path in ['w_z_data_coll']:
       plot(path)