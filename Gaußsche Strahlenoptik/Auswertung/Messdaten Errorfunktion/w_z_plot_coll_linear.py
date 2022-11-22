# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 16:23:03 2022

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

def Regr(z,w0,z0):
    lamb = .0006328
    zR = np.pi*w0**2/lamb
    return -(w0/zR*z-z0*w0/zR)

def Regr_sp(tup,z):
    return Regr(z,*tup)

def LineIntersection(m1,m2,b1,b2):
    """determining the geometric center of the 1-sigma-range"""  
    x = (b2-b1)/(m1-m2)
    y = m1*x+b1
    return x,y

def LinRegr(tup, x):
    """Model of a linear function for fitting and plotting"""
    a,b = tup
    return Regr(x,a,b)

def plot(CSVNAME):
    PATH = CSVNAME + '.csv'


    p0 = (1,100)

    X, Y, Yerr = np.loadtxt(PATH,delimiter=',')    
      
    
    Xerr = np.ones(len(X))*1
    """
    ind = 2
    X = np.delete(X,ind)
    Y = np.delete(Y,ind)
    Xerr = np.delete(Xerr,ind)
    Yerr = np.delete(Yerr,ind)
    print(X)
    """
    xlabel = r'$\frac{z}{\mathrm{mm}}$' 
    ylabel = r'$\frac{w(z)}{\mathrm{mm}}$'
    
    xlim = (-100,400)
    ylim = (min(Y)*0.9-.1,max(Y)*1.1)

    popt,pcov = curve_fit(Regr,X,Y,p0=p0,maxfev=100000,sigma=Yerr,absolute_sigma=True)
    stdDev=np.sqrt(np.diag(pcov))  


    t1 = np.linspace(xlim[0],xlim[1], 10**4)

    fig,ax=plt.subplots(1,1,figsize=(10,10/np.sqrt(2))) 
  
    ErrDiag11 = LinRegr((popt[0]+stdDev[0],popt[1]-stdDev[1]),t1) #maximal slope a, minimal y intercept b
    ErrDiag21 = LinRegr((popt[0]-stdDev[0],popt[1]+stdDev[1]),t1) #minimal slope a, maximal y intercept b

    ErrAr_top11 = [t1[-1],ErrDiag11[-1]]
    ErrAr_top21 = [t1[0],ErrDiag21[0]]
    ErrAr_bot11 = [t1[-1],ErrDiag21[-1]]
    ErrAr_bot21 = [t1[0],ErrDiag11[0]]
    ErrAr_top1 = (ErrAr_top11[1]-ErrAr_top21[1])/(ErrAr_top11[0]-ErrAr_top21[0])*(t1-ErrAr_top21[0])+ErrAr_top21[1]
    ErrAr_bottom1 = (ErrAr_bot11[1]-ErrAr_bot21[1])/(ErrAr_bot11[0]-ErrAr_bot21[0])*(t1-ErrAr_bot21[0])+ErrAr_bot21[1]
    centerX, centerY = LineIntersection(popt[0]+stdDev[0],popt[0]-stdDev[0],popt[1]-stdDev[1],popt[1]+stdDev[1])

   #------Adding the fitted lines and the measurements to figure and axis--------
    ax.plot(t1,LinRegr(popt,t1), marker = 'None', linestyle = '-', label=r'$w(z)=\frac{w_0}{z_R}(z_R-z)$') #optimized line
    ax.plot(t1,ErrDiag11, marker = 'None', linestyle = 'dashed') #deviating line 1 crossing the 1sigma-range
    ax.plot(t1,ErrDiag21, marker = 'None', linestyle = 'dashed') #deviating line 2 crossing the 1sigma-range
    ax.errorbar(x=X,y=Y,xerr=Xerr,yerr=Yerr,marker='x',linestyle='None',label='Messwerte')
    ax.fill_between(t1,ErrAr_bottom1,ErrAr_top1, alpha = .6, label = r'$1\sigma-\text{Fehlerstreifen}$') #Area containing ~33% of all points of the set (1sigma-range) 
    ax.scatter(centerX,centerY,marker='v', label = 'Schwerpunkt', color='maroon', edgecolor='black') #geometric center of 1sigma-range
    
    
    sigfig = 4
    stdDev_rounded = ['{:g}'.format(float('{:.{p}g}'.format(stdDev[i], p=sigfig))) for i in range(0,len(popt))]
    decimals = [len(str(stdDev_rounded[i].split('.')[-1])) for i in range(0,len(popt))]
    cell_text = [[str(round(popt[i],decimals[i])),str(round(stdDev[i],decimals[i]))] for i in range(0,len(popt))]
    
    the_table = the_table = plt.table(cellText=cell_text,
                      rowLabels=[r'$w_0[\mathrm{mm}]$',r'$z_0[\mathrm{mm}]$'],
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
    plt.savefig(CSVNAME+'_linear'+".pdf",dpi=1200)
    plt.savefig(CSVNAME+'_linear'+".png",dpi=1200)



if __name__ == "__main__":    
   for path in ['w_z_data_coll']:
       plot(path)