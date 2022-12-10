# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 11:27:48 2022

@author: Jan-Philipp
"""

import numpy as np #math functions, arrays
import matplotlib.pyplot as plt #visualizing
from scipy.interpolate import make_interp_spline
import matplotlib
import pandas as pd
from scipy.optimize import curve_fit

matplotlib.style.use('JaPh') #merely contains basic style information
plt.ioff()

def Fit(x,a,b,c,d):
    return np.tanh(a*(x-b))*c+d

def plot(CSVNAME):
    global popt1
    PATH = CSVNAME + '.csv'
    fig,ax=plt.subplots(1,1,figsize=(10,10/np.sqrt(2))) 
    
    df = pd.read_csv(PATH,sep=',',header=None)
    Y1=df.iloc[:,10]
    Y2=df.iloc[:,11]  
    X =df.iloc[:,6]
    print(Y1,Y2,X)
    X = np.log(X)
    Y1 = np.log(Y1)
    Y2 = np.log(Y2)
    
    ylabel = r'$\mathrm{ln}\left(\frac{G^\prime}{\mathrm{Pa}}\right), \mathrm{ln}\left(\frac{G^{\prime\prime}}{\mathrm{Pa}}\right)$' 
    xlabel = r'$\mathrm{ln}\left(\frac{f}{\mathrm{Hz}}\right)$'
    
    Ymax = np.max(np.array([Y1,Y2]),axis=0)
    Ymin = np.min(np.array([Y1,Y2]),axis=0)

    dx = (max(X)-min(X))/10
    dy = (max(Ymax)-min(Ymin))/15
    xlim = (min(X)-dx,max(X)+dx)
    ylim = (min(Ymin)-dy,max(Ymax)+dy)
    
    #X_Y1_Spline = make_interp_spline(X, Y1,k=3)
    #X_Y2_Spline = make_interp_spline(X, Y2,k=3)
    
    popt1,pcov1 = curve_fit(Fit,X,Y1,maxfev = 100000,p0=[0.79769266, 3.77394899, 5.19260498, 7.45366197])
    popt2,pcov2 = curve_fit(Fit,X,Y2,maxfev = 100000,p0=[0.79769266, 3.77394899, 5.19260498, 7.45366197])
    
    t1 = np.linspace(min(X),max(X), 10**3)
    
    minind1 = np.argmin(np.round(np.abs(Fit(t1,*popt2)-Fit(t1,*popt1)),1))
    minind2 = np.argmin(np.round(np.abs(Fit(t1,*popt2)-Fit(t1,*popt1)),4))
    
    
    #ax.plot(t1,Fit(t1,*popt2),marker='None',linestyle='-',color='red')
    #ax.plot(t1,Fit(t1,*popt1),marker='None',linestyle='-',color='blue')
    #ax.plot(t1,X_Y1_Spline(t1),marker='None',color='black',linestyle='-')
    #ax.plot(t1,X_Y2_Spline(t1),marker='None',color='black',linestyle='-')
    ax.plot(X,Y2,marker='x',linestyle='None',color='red',label=r'$\mathrm{ln}\left(\frac{G^{{\prime\prime}}}{\mathrm{Pa}}\right)$')
    ax.plot(X,Y1,marker='x',linestyle='None',color='blue',label=r'$\mathrm{ln}\left(\frac{G^{\prime}}{\mathrm{Pa}}\right)$')
   
    
    #ax.axvline((t1[minind2]+t1[minind1])/2,marker='None',linestyle='dotted',color='navajowhite',label=r'Übergangsfrequenz $f^*=\left(%.2f\pm%.2f\right)\ \mathrm{Hz}$'%(np.exp((t1[minind2]+t1[minind1])/2),(np.exp(t1[minind2])-np.exp(t1[minind1]))/2))
    #ax.axvline(t1[minind2],marker='None',linestyle='dotted',color='navajowhite')
    #ax.axvspan(t1[minind1],t1[minind2],linestyle='None',alpha=.2,color='navajowhite',label=r'$\mathrm{ln}\left(\frac{G^{{\prime\prime}}}{\mathrm{Pa}}\right)\approx \mathrm{ln}\left(\frac{G^{\prime}}{\mathrm{Pa}}\right)$')
    ax.axvspan(-10,-1,linestyle='None',alpha=.2,color='lightgreen',label=r'$\left(\mathrm{ln}\left(\frac{G^{{\prime\prime}}}{\mathrm{Pa}}\right)- \mathrm{ln}\left(\frac{G^{\prime}}{\mathrm{Pa}}\right)\right)\approx \mathrm{const.}<0$')
    ax.axvspan(-1,(t1[minind2]+t1[minind1])/2,linestyle='None',alpha=.2,color='lightcoral',label=r'$\mathrm{ln}\left(\frac{G^{{\prime\prime}}}{\mathrm{Pa}}\right)< \mathrm{ln}\left(\frac{G^{\prime}}{\mathrm{Pa}}\right)$')
    ax.axvspan((t1[minind2]+t1[minind1])/2,4.3,linestyle='None',alpha=.2,color='lightskyblue',label=r'$\mathrm{ln}\left(\frac{G^{{\prime\prime}}}{\mathrm{Pa}}\right)> \mathrm{ln}\left(\frac{G^{\prime}}{\mathrm{Pa}}\right)$')
    ax.axvspan(4.3,10,linestyle='None',alpha=.2,color='midnightblue',label=r'$\mathrm{ln}\left(\frac{G^{{\prime\prime}}}{\mathrm{Pa}}\right)< \mathrm{ln}\left(\frac{G^{\prime}}{\mathrm{Pa}}\right)$')
    #ax.annotate(r'$c^*$',(popt[0],(popt[1]+min(Y))/2),xytext=((popt[0]*1.5+max(X)*.5)/(2),(popt[1]+min(Y))/2),arrowprops=dict(arrowstyle= matplotlib.patches.ArrowStyle("Fancy", head_length=.2, head_width=.2, tail_width=.1),color='black'))
    
    
    ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlabel(xlabel,size=16)
    ax.set_ylabel(ylabel,size=16)
    ax.set_xlim(xlim[0],xlim[1])
    ax.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)

    ax.legend(loc='best',fontsize=15)
    fig.tight_layout()

    #----Save Figure-----------
    plt.savefig("Osc_Guaran.pdf",dpi=1200)
        
if __name__ == "__main__":
    plot('Osc_Guaran')