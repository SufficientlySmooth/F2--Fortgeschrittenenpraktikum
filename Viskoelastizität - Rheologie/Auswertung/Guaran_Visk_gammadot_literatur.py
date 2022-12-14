# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 13:43:06 2022

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


def plot():
   
    fig,ax=plt.subplots(1,1,figsize=(10,10/np.sqrt(2))) 
    
    colorlist = ['forestgreen','maroon','orange','darkgray','midnightblue','darkmagenta','darkgoldenrod','darkslategray','red','lavender','magenta']
    
    shear_ind = 2
    cs = [1.45,1,.5,.25]
    cs = np.array(cs)
    maxy = -np.inf
    miny = np.inf
    stdList = []
    poptList = []
    markers = ['o','x','.','^','.','+','d']
    ylabel = r'$\mathrm{log}_{10}\left( \dfrac{\eta}{\mathrm{Pa}\cdot s}\right)$' 
    xlabel = r'$\mathrm{log}_{10}\left(\frac{\dot\gamma}{s^{-1}}\right)$'
    
    imp = pd.read_csv('Guaran_Visk_gammadot_literatur.csv',sep=';',header=None)
    Xs = [np.array(imp.iloc[2*i,:]) for i in range(0,4)]
    Ys = [np.array(imp.iloc[2*i+1,:]) for i in range(0,4)]


    for i in range(0,len(Xs)):

        y=Ys[i]
        x=Xs[i]

        x = x[~np.isnan(x)]
        y = y[~np.isnan(y)]

        xerr_rel= xerr_abs= yerr_rel= yerr_abs = np.zeros(len(x))
        xerr = x*xerr_rel+xerr_abs
        yerr = y*yerr_rel+yerr_abs


        X,Y = LinXY(x,y)
        Xerr,Yerr = LinXYerr(xerr,yerr,x,y)
        
        maxy = max(Y) if maxy<max(Y) else maxy
        miny = min(Y) if miny>min(Y) else miny
        
        dx = (max(X)-min(X))/10
        dy = (maxy-miny)/15
        xlim = (min(X)-dx,max(X)+dx)
        ylim = (miny-dy,maxy+dy)


        popt, pcov = curve_fit(LinRegr1,X,Y,maxfev=10000)
        stdDev=np.sqrt(np.diag(pcov))  

        t1 = np.linspace(xlim[0],xlim[1], 500)

    

        #ax.axvline(popt[0],marker='None',linestyle='dotted',color='navajowhite',label=r'$c^*$')
        #ax.axvspan(popt[0]-stdDev[0],popt[0]+stdDev[0],linestyle='None',alpha=.2,color='navajowhite')
        #ax.annotate(r'$c^*$',(popt[0],(popt[1]+min(Y))/2),xytext=((popt[0]*1.5+max(X)*.5)/(2),(popt[1]+min(Y))/2),arrowprops=dict(arrowstyle= matplotlib.patches.ArrowStyle("Fancy", head_length=.2, head_width=.2, tail_width=.1),color='black'))
        
        
        if i == 0:
            ax.plot(t1,LinRegr(popt,t1), marker = 'None', linestyle = '-', color = colorlist[i], label=ylabel[:-1]+'=\\alpha \cdot'+xlabel[1:-1]+'+\\beta$')
        else:
            ax.plot(t1,LinRegr(popt,t1), marker = 'None', linestyle = '-', color = colorlist[i])
        ax.plot(X,Y,marker=markers[i], color = colorlist[i],linestyle='None',label=r'Messwerte f??r $c='+str(np.round(cs[i],2))+'\%$')
        poptList.append(popt)
        stdList.append(stdDev)

    cell_text = np.core.defchararray.add(np.core.defchararray.add(np.round(np.transpose(poptList)).astype(str),np.full(np.transpose(poptList).shape,'+-')), np.round(np.transpose(stdList)).astype(str))
    #cell_text = np.core.defchararray.add(np.core.defchararray.add(np.full(np.full(np.transpose(poptList).shape,'$'),cell_text)),np.full(np.transpose(poptList).shape,'$'))
    sigfig = 2
    stdDevFlat = np.array(stdList).flatten()
    poptFlat = np.array(poptList).flatten()
   
    stdDev_rounded = ['{:g}'.format(float('{:.{p}g}'.format(stdDevFlat[i], p=sigfig)))+'$' for i in range(0,len(stdDevFlat))]
    decimals = [len(str(stdDev_rounded[i].split('.')[-1])) for i in range(0,len(stdDev_rounded))]
    popt_rounded = [r'$'+str(round(poptFlat[i],decimals[i]))+'\pm' for i in range(len(poptFlat))]
    poptList = np.reshape(np.array(popt_rounded),np.array(poptList).shape)
    stdList = np.reshape(np.array(stdDev_rounded),np.array(stdList).shape)
    cell_text = np.core.defchararray.add(np.transpose(poptList), np.transpose(stdList).astype(str))
    
    the_table = the_table = plt.table(cellText=cell_text,
                    #rowLabels = np.full(7,'a'),
                      rowLabels=[r'$\alpha$',r'$\beta$'],
                      colLabels=[r'$c='+str(np.round(i,2))+'\% $' for i in cs],
                      loc='bottom',
                      cellLoc='center',
                      bbox=[0,-0.31,1,0.2],
                      edges='closed')
    #the_table.auto_set_font_size(False)
    #the_table.set_fontsize(8)
    
    ax.set_ylim(ylim[0],maxy)
    ax.set_xlabel(xlabel,size=16)
    ax.set_ylabel(ylabel,size=16)
    ax.set_xlim(xlim[0],xlim[1])
    ax.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)

    ax.legend(loc='best',fontsize=11,ncol=2,columnspacing=0.6,labelspacing=.3)
    
    
    #fig.tight_layout()

    #----Save Figure-----------
    plt.savefig("Guar_Viscosity_vs_Shearrate_literature.pdf",dpi=1200)
    
        
if __name__ == "__main__":
    plot()