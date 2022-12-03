# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:49:25 2022

@author: Jan-Philipp
"""

import numpy as np #math functions, arrays
import matplotlib.pyplot as plt #visualizing
from scipy.optimize import curve_fit
from itertools import product, combinations
import matplotlib
import pandas as pd

matplotlib.style.use('JaPh') #merely contains basic style information
plt.ioff()


def LinXY(x,y):
    """"Scale x and y to obtain an linear relation between ordinate and abscissa"""
    return x,y


def LinXYerr(xerr,yerr,x,y):
    """"Scale xerr and yerr according to LinXY"""
    return xerr,yerr

def LinRegr1(x, x0, y0, k1, k2):
    """Model of a piecewise linear function for fitting and plotting"""
    tup = (x0, y0, k1, k2)
    return LinRegr(tup, x)

def LinRegr(tup, x):
    """Model of a piecewise linear function for fitting and plotting"""
    x0, y0, k1, k2 = tup
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

def LineIntersection(m1,m2,b1,b2):
    """determining the geometric center of the 1-sigma-range"""  
    x = (b2-b1)/(m1-m2)
    y = m1*x+b1
    return x,y

def plot(CSVNAMES):
    
   
    fig,ax=plt.subplots(1,1,figsize=(10,10/np.sqrt(2))) 
    
    
    shear_ind = 2
    x = [0,.25,.5,1,1.4,2,2.3]
    y = []
    for CSVNAME in CSVNAMES:
        PATH = CSVNAME + '.csv'
        df = pd.read_csv(PATH,sep=',',header=None)
        y.append(df.iloc[shear_ind,8])
        
    x = np.array(x)/100
    y = np.array(y)
    print(y)
    
    xerr_rel= xerr_abs= yerr_rel= yerr_abs = np.zeros(len(x))
    xerr = x*xerr_rel+xerr_abs
    yerr = y*yerr_rel+yerr_abs
    
    ylabel = r'$\dfrac{\eta}{\mathrm{Pa}\cdot s}$' 
    xlabel = r'$c$'
    
    X,Y = LinXY(x,y)
    Xerr,Yerr = LinXYerr(xerr,yerr,x,y)
    dx = (max(X)-min(X))/10
    dy = (max(Y)-min(Y))/15
    xlim = (min(X)-dx,max(X)+dx)
    ylim = (min(Y)-dy,max(Y)+dy)
    
    
    popt, pcov = curve_fit(LinRegr1,X,Y,p0=(1.3/100,5,2/.01,35/.01),maxfev=10000)
    stdDev=np.sqrt(np.diag(pcov))  
    
    t1 = np.linspace(xlim[0],xlim[1], 10**3)
    

    ax.plot(t1,LinRegr(popt,t1), marker = 'None', linestyle = '-', label=ylabel[:-1]+'=\\begin{cases} c\leq c^*: & [\eta ] \cdot '+xlabel[1:-1]+'+\eta_{c^*}-[\eta ]\cdot c^*  \\\ c> c^*: & [\eta ]^\prime \cdot '+xlabel[1:-1]+'+\eta_{c^*}-[\eta ]^\prime\cdot c^*  \end{cases}$') #optimized line
    ax.errorbar(x=X,y=Y,xerr=Xerr,yerr=Yerr,marker='x',linestyle='None',label='Messwerte')
    
    ax.axvline(popt[0],marker='None',linestyle='dotted',color='navajowhite',label=r'$c^*$')
    ax.axvspan(popt[0]-stdDev[0],popt[0]+stdDev[0],linestyle='None',alpha=.2,color='navajowhite')
    #ax.annotate(r'$c^*$',(popt[0],(popt[1]+min(Y))/2),xytext=((popt[0]*1.5+max(X)*.5)/(2),(popt[1]+min(Y))/2),arrowprops=dict(arrowstyle= matplotlib.patches.ArrowStyle("Fancy", head_length=.2, head_width=.2, tail_width=.1),color='black'))
    
    PointWise_LowerBound = np.zeros(len(t1))
    PointWise_UpperBound = np.zeros(len(t1))
    for tind in range(len(t1)):
        t = t1[tind]
        Min = LinRegr(popt,t)
        Max = LinRegr(popt,t)
        for comb1, comb2 in combinations(product([-1,1],repeat=len(stdDev)),2):
            p1 = popt+stdDev*comb1
            p2 = popt+stdDev*comb2
            Min = min(LinRegr(p1,t),Min)
            Max = max(LinRegr(p2,t),Max)
        PointWise_LowerBound[tind] = Min
        PointWise_UpperBound[tind] = Max
    
    ax.fill_between(t1,PointWise_LowerBound,PointWise_UpperBound, alpha = .6, label = r'$1\sigma-\text{Fehlerstreifen}$')
            
    sigfig = 4
    stdDev_rounded = ['{:g}'.format(float('{:.{p}g}'.format(stdDev[i], p=sigfig))) for i in range(0,len(popt))]
    decimals = [len(str(stdDev_rounded[i].split('.')[-1])) for i in range(0,len(popt))]
    cell_text = [[str(round(popt[i],decimals[i])),str(round(stdDev[i],decimals[i]))] for i in range(0,len(popt))]
    
    the_table = the_table = plt.table(cellText=cell_text,
                      rowLabels=[r'$c^*$',r'$\eta_{c^*}$',r'$[\eta ]$',r'$[\eta ]^\prime$'],
                      colLabels=['Wert','Unsicherheit'],
                      loc='bottom',
                      cellLoc='center',
                      bbox=[0.25,-0.31,0.5,0.2],
                      edges='closed')

    
    ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlabel(xlabel,size=16)
    ax.set_ylabel(ylabel,size=16)
    ax.set_xlim(xlim[0],xlim[1])
    ax.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)

    ax.legend(loc='best',fontsize=15)
    fig.tight_layout()

    #----Save Figure-----------
    plt.savefig("Viscosity_by_Concentration_Guar_per_Shearrate.pdf",dpi=1200)
        
if __name__ == "__main__":
    plot(['guar_00000_percent','guar_00025_percent','guar_00050_percent','guar_00100_percent','guar_00140_percent','guar_00200_percent','guar_00230_percent'])