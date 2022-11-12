# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:45:24 2022

@author: Jan-Philipp
"""

import numpy as np #math functions, arrays
import matplotlib.pyplot as plt #visualizing
from scipy.optimize import curve_fit #curv fitting
import matplotlib
from scipy.special import erf
from itertools import product, combinations

matplotlib.style.use('JaPh') #merely contains basic style information
plt.ioff()

def Regr(x,a,b,c):
    """Model of a linear function for fitting and plotting"""
    return a*(erf(np.sqrt(2)/b*(x-c))+1)
    
def plot(PDFname):
    global x,y
    fig,ax=plt.subplots(1,1,figsize=(10,10/np.sqrt(2))) #create axis embedded in figure
    
    X = np.linspace(0,10,20)
    Y = Regr(X,3,4,5)*np.random.uniform(0.9,1.1,size=20)
       
    Yerr = np.sqrt(Y)*.1
    xlabel = r'$\frac{x}{\mathrm{cm}}$' 
    ylabel = r'$\frac{P(x)}{\mathrm{mW}}$'
    
    xlim = (min(X),max(X))
    ylim = (min(Y),max(Y)*1.1)

    popt,pcov = curve_fit(Regr,X,Y,sigma=Yerr,p0=(3,4,5),maxfev=100000)
    stdDev=np.sqrt(np.diag(pcov))  #extracting the standard deviation of a and b from pcov (diagonally)
        
    t1 = np.linspace(xlim[0],xlim[1], 10**4)

   #------Adding the fitted lines and the measurements to figure and axis--------
    ax.plot(t1,Regr(t1,*popt), marker = 'None', linestyle = '-',label=r'$P(x)=P_0\cdot\left(\mathrm{erf}\left(\frac{\sqrt{2}}{w}(x-x_0)\right)+1\right)$') #optimized line
    ax.errorbar(X,Y,yerr=Yerr,marker='x',linestyle='None',label='Messwerte')
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
    print(decimals)
    print(stdDev_rounded)
    cell_text = [[str(round(popt[i],decimals[i])),str(round(stdDev[i],decimals[i]))] for i in range(0,len(popt))]
    the_table = the_table = plt.table(cellText=cell_text,
                      rowLabels=[r'$P_0[\mathrm{mW}]$',r'$w[\mathrm m]$',r'$x_0[\mathrm m]$'],
                      #rowColours=colors,
                      colLabels=['Wert','Unsicherheit'],
                      loc='bottom',
                      cellLoc='center',
                      bbox=[0.25,-0.3,0.5,0.2],
                      #AXESPAD = 0.1,
                      edges='closed')
    #ax.bbox.get_points()

    #-----Formatting------------
    ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlabel(xlabel,size=16)
    ax.set_ylabel(ylabel,size=16)
    ax.set_xlim(xlim[0],xlim[1])
    ax.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)
    #ax.legend(loc='upper center',fontsize=15,bbox_to_anchor=(.8, -0.13),ncol=1)
    ax.legend(loc='best',fontsize=15)
    #ax.legend(loc='best',fontsize=15,bbox_to_anchor=(0.5,-0.2,0.6,0.2),ncol=2)
    fig.tight_layout()

    #----Save Figure-----------
    plt.savefig(PDFname+".pdf",dpi=1200)
    plt.savefig(PDFname+".png",dpi=1200)
                
if __name__ == "__main__":
    plot('erf')