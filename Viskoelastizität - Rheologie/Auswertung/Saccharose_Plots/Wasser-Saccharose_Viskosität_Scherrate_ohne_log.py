# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:49:25 2022

@author: Jan-Philipp
"""

import numpy as np #math functions, arrays
import matplotlib.pyplot as plt #visualizing
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib

matplotlib.style.use('JaPh') #merely contains basic style information
plt.ioff()


def LinXY(x,y):
    """"Scale x and y to obtain an linear relation between ordinate and abscissa"""
    return x,y

def LinXYerr(xerr,yerr,x,y):
    """"Scale xerr and yerr according to LinXY"""
    return xerr/np.abs(x),yerr/np.abs(y)

def LinRegr1(x,a,b,c,d):
    """Model of a linear function for fitting and plotting"""
    return a*x**b+c*x**d

def LinRegr(tup, x):
    """Model of a linear function for fitting and plotting"""
    return LinRegr1(x,*tup)

def LineIntersection(m1,m2,b1,b2):
    """determining the geometric center of the 1-sigma-range"""  
    x = (b2-b1)/(m1-m2)
    y = m1*x+b1
    return x,y

def plot(CSVNAME):
    
    PATH = CSVNAME + '.csv'
    
    fig,ax=plt.subplots(1,1,figsize=(10,10/np.sqrt(2))) #create axis embedded in figure
    
    df = pd.read_csv(PATH,sep=',',header=None)
    
    x = np.array(df.iloc[:,7])
    y = np.array(df.iloc[:,8])
   
    ylabel = r'$ \dfrac{\eta}{\mathrm{Pa}\cdot s}$' 
    xlabel = r'$\frac{\dot\gamma}{s^{-1}}$'
    
    X,Y = LinXY(x,y)
    dx = (max(X)-min(X))/10
    dy = (max(Y)-min(Y))/15
    xlim = (min(X)-dx,max(X)+dx)
    ylim = (min(Y)-dy,max(Y)+dy)
    
    
    popt, pcov = curve_fit(LinRegr1,X,Y,maxfev=10000)
    stdDev=np.sqrt(np.diag(pcov))  #extracting the standard deviation of a and b from pcov (diagonally)
    
    #------setting up the deviating lines (diagonals in 1sigma-range) --------
    t_for_sigma = np.linspace(min(X),max(X),10000)
    t1 = np.linspace(xlim[0],xlim[1], 10**4)
    
   #------Adding the fitted lines and the measurements to figure and axis--------
    ax.plot(t1,LinRegr(popt,t1), marker = 'None', linestyle = '-', label=ylabel[:-1]+'=\\alpha \cdot'+xlabel[1:-1]+'+\\beta$') #optimized line
   
    ax.scatter(X,Y,marker='x',linestyle='None',label='Messwerte')
   
    """
    stdDev = list(stdDev)
    popt = list(popt)
    stdDev.append(np.mean(y))
    popt.append(np.std(y))
    """
    sigfig = 4
    stdDev_rounded = ['{:g}'.format(float('{:.{p}g}'.format(stdDev[i], p=sigfig))) for i in range(0,len(popt))]
    decimals = [len(str(stdDev_rounded[i].split('.')[-1])) for i in range(0,len(popt))]
    cell_text = [[str(round(popt[i],decimals[i])),str(round(stdDev[i],decimals[i]))] for i in range(0,len(popt))]

    the_table = the_table = plt.table(cellText=cell_text,
                      rowLabels=[r'$\alpha$',r'$\beta$',r'$\gamma$',r'$\gamma$'],
                      colLabels=['Wert','Unsicherheit'],
                      loc='bottom',
                      cellLoc='center',
                      bbox=[0.25,-0.33,0.5,0.2],
                      edges='closed')

    ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlabel(xlabel,size=16)
    ax.set_ylabel(ylabel,size=16)
    ax.set_xlim(xlim[0],xlim[1])
    ax.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)

    ax.legend(loc='best',fontsize=15)
    fig.tight_layout()

    #----Save Figure-----------
    plt.savefig(CSVNAME+"_visc_vs_shear_rate_non_log.pdf",dpi=1200)
        
if __name__ == "__main__":
    for path in ['sac_00_percent','sac_10_percent','sac_20_percent','sac_40_percent']:
        plot(path)