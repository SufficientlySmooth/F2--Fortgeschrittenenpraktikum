# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 11:14:25 2022

@author: Jan-Philipp
"""

import numpy as np #math functions, arrays
import matplotlib.pyplot as plt #visualizing
from scipy.optimize import curve_fit
import matplotlib

matplotlib.style.use('JaPh') #merely contains basic style information
plt.ioff()


def LinXY(x,y):
    """"Scale x and y to obtain an linear relation between ordinate and abscissa"""
    return np.log(x),np.log(y)

def LinXYerr(xerr,yerr,x,y):
    """"Scale xerr and yerr according to LinXY"""
    return xerr/np.abs(x),yerr/np.abs(y)

def LinRegr1(x,a,b):
    """Model of a linear function for fitting and plotting"""
    return a*x+b

def LinRegr(tup, x):
    """Model of a linear function for fitting and plotting"""
    a,b = tup
    return a*x+b

def LineIntersection(m1,m2,b1,b2):
    """determining the geometric center of the 1-sigma-range"""  
    x = (b2-b1)/(m1-m2)
    y = m1*x+b1
    return x,y

def plot(CSVNAME):
    
    PATH = CSVNAME + '.csv'
    
    fig,ax=plt.subplots(1,1,figsize=(10,10/np.sqrt(2))) #create axis embedded in figure
    
    
    x, y, xerr_rel, xerr_abs, yerr_rel, yerr_abs = np.loadtxt(PATH,delimiter=';',skiprows=1,unpack=True)    
            
    xerr = x*xerr_rel+xerr_abs
    yerr = y*yerr_rel+yerr_abs
    
    ylabel = r'$\mathrm{ln}\left( \dfrac{\sigma}{\mathrm{Pa}}\right)$' 
    xlabel = r'$\mathrm{ln}\left(\frac{\dot\gamma}{s^{-1}}\right)$'
    
    X,Y = LinXY(x,y)
    Xerr,Yerr = LinXYerr(xerr,yerr,x,y)
    xlim = (.9*min(X),1.1*max(X))
    ylim = (min(Y)*.9,max(Y)*1.1)
    
    
    popt, pcov = curve_fit(LinRegr1,X,Y,sigma=Yerr,absolute_sigma=True)
    stdDev=np.sqrt(np.diag(pcov))  #extracting the standard deviation of a and b from pcov (diagonally)
    
    #------setting up the deviating lines (diagonals in 1sigma-range) --------
    t_for_sigma = np.linspace(min(X),max(X),10000)
    t1 = np.linspace(xlim[0],xlim[1], 10**4)
    ErrDiag11 = LinRegr((popt[0]+stdDev[0],popt[1]-stdDev[1]),t_for_sigma) #maximal slope a, minimal y intercept b
    ErrDiag21 = LinRegr((popt[0]-stdDev[0],popt[1]+stdDev[1]),t_for_sigma) #minimal slope a, maximal y intercept b

    ErrAr_top11 = [t1[-1],ErrDiag11[-1]]
    ErrAr_top21 = [t1[0],ErrDiag21[0]]
    ErrAr_bot11 = [t1[-1],ErrDiag21[-1]]
    ErrAr_bot21 = [t1[0],ErrDiag11[0]]
    ErrAr_top1 = (ErrAr_top11[1]-ErrAr_top21[1])/(ErrAr_top11[0]-ErrAr_top21[0])*(t1-ErrAr_top21[0])+ErrAr_top21[1]
    ErrAr_bottom1 = (ErrAr_bot11[1]-ErrAr_bot21[1])/(ErrAr_bot11[0]-ErrAr_bot21[0])*(t1-ErrAr_bot21[0])+ErrAr_bot21[1]
    centerX, centerY = LineIntersection(popt[0]+stdDev[0],popt[0]-stdDev[0],popt[1]-stdDev[1],popt[1]+stdDev[1])

   #------Adding the fitted lines and the measurements to figure and axis--------
    ax.plot(t1,LinRegr(popt,t1), marker = 'None', linestyle = '-', label=ylabel[:-1]+'_{opt.}=\\alpha \cdot'+xlabel[1:-1]+'+\\beta$') #optimized line
    ax.plot(t_for_sigma,ErrDiag11, marker = 'None', linestyle = 'dashed') #deviating line 1 crossing the 1sigma-range
    ax.plot(t_for_sigma,ErrDiag21, marker = 'None', linestyle = 'dashed') #deviating line 2 crossing the 1sigma-range
    ax.errorbar(x=X,y=Y,xerr=Xerr,yerr=Yerr,marker='x',linestyle='None',label='Messwerte')
    ax.fill_between(t_for_sigma,ErrAr_bottom1,ErrAr_top1, alpha = .6, label = r'$1\sigma-\text{Fehlerstreifen}$') #Area containing ~33% of all points of the set (1sigma-range) 
    ax.scatter(centerX,centerY,marker='v', label = 'Schwerpunkt', color='maroon', edgecolor='black') #geometric center of 1sigma-range
    
    sigfig = 4
    stdDev_rounded = ['{:g}'.format(float('{:.{p}g}'.format(stdDev[i], p=sigfig))) for i in range(0,len(popt))]
    decimals = [len(str(stdDev_rounded[i].split('.')[-1])) for i in range(0,len(popt))]
    cell_text = [[str(round(popt[i],decimals[i])),str(round(stdDev[i],decimals[i]))] for i in range(0,len(popt))]
    
    the_table = the_table = plt.table(cellText=cell_text,
                      rowLabels=[r'$\alpha$',r'$\beta$'],
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
    plt.savefig(CSVNAME+".pdf",dpi=1200)
        
if __name__ == "__main__":
    for path in ['Test_plot_2']:
        plot(path)