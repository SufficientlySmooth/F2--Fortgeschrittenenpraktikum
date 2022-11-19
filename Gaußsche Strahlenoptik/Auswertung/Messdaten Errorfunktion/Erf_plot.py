# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 11:47:29 2022

@author: Jan-Philipp
"""
import os    
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit 
import matplotlib
from scipy.special import erf
from itertools import product, combinations

matplotlib.style.use('JaPh') 
plt.ioff()

def Regr(x,a,b,c):
    return a*(erf(-np.sqrt(2)/b*(x-c))+1)
    
def plot(FILENAME):
    PATH = FILENAME+'.csv'
    
    X,Xerr,Y,Yerr = np.loadtxt(PATH,delimiter=';',skiprows=1,unpack=True)
    Y-=min(Y)
    Yerr = np.sqrt(Y)*.1
    xlabel = r'$\frac{x}{\mathrm{mm}}$' 
    ylabel = r'$\frac{U}{\mathrm{V}}$'
    
    xlim = (-.5+min(X),max(X)+.5)
    ylim = (0,max(Y)*1.1)

    popt,pcov = curve_fit(Regr,X,Y,p0=(max(Y)/2,np.mean(X),1),maxfev=1000000)
    stdDev=np.sqrt(np.diag(pcov))  
        
    t1 = np.linspace(xlim[0],xlim[1], 10**4)
    
    fig,ax=plt.subplots(1,1,figsize=(10,10/np.sqrt(2))) 

    ax.plot(t1,Regr(t1,*popt), marker = 'None', linestyle = '-',label=r'$U(x)=U_0\cdot\left(\mathrm{erf}\left(\frac{\sqrt{2}}{w}(x_0-x)\right)+1\right)$') 
    ax.errorbar(X,Y,yerr=Yerr,xerr=Xerr,marker='x',linestyle='None',label='Messwerte gem. \emph{'+PATH+'}')
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
                      rowLabels=[r'$U_0[\mathrm{V}]$',r'$w[\mathrm mm]$',r'$x_0[\mathrm mm]$'],
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

    plt.savefig(FILENAME+".pdf",dpi=1200)
    plt.savefig(FILENAME+".png",dpi=1200)
    
    return popt[1],stdDev[1]
    

if __name__ == "__main__":    
    w_coll = []
    dw_coll = []
    w_div = []
    dw_div = []
    z_coll = []
    z_div = []
    path = os.getcwd() + "\\"
    dirs = os.listdir( path )
    for item in dirs:
        if os.path.isfile(path+item) and item[::-1][:3][::-1]=='csv' and item[0] in ['C','D']:
            f, e = os.path.splitext(item)
            w,dw = plot(f)
            z = float(f.split('_')[-1])
            if item[0]=='C':
                w_coll.append(w)
                dw_coll.append(dw)
                z_coll.append(z)
            elif item[0]=='D':
                w_div.append(w)
                dw_div.append(dw)
                z_div.append(z)
    np.savetxt('w_z_data_coll.csv',[z_coll,w_coll,dw_coll],delimiter=',')
    np.savetxt('w_z_data_div.csv',[z_div,w_div,dw_div],delimiter=',')