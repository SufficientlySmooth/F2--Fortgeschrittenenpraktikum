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

def plot(CSVNAMES):
    
    colorlist = ['darkgreen','maroon','darkgray','midnightblue','darkmagenta','darkgoldenrod','darkslategray','red','lavender','orange','magenta']
    
    fig,ax=plt.subplots(1,1,figsize=(10,10/np.sqrt(2))) #create axis embedded in figure
    

    x = [0,10,20]
    x = np.array(x)/100
    
    stdList = []
    poptList = []
    gammadotList = []
    
    maxy = -np.inf
    for shear_ind in range(0,11):
        print(shear_ind)
        y = []
        for CSVNAME in CSVNAMES:
            PATH = CSVNAME + '.csv'
            df = pd.read_csv(PATH,sep=',',header=None)
            y.append(df.iloc[shear_ind,8])
        
        y = np.array(y)
        #print(y)
    
        maxy = max(y) if maxy<max(y) else maxy
   
        xerr_rel= xerr_abs= yerr_rel= yerr_abs = np.zeros(len(x))
        xerr = x*xerr_rel+xerr_abs
        yerr = y*yerr_rel+yerr_abs
    
        ylabel = r'$\dfrac{\eta}{\mathrm{Pa}\cdot s}$' 
        xlabel = r'$c$'
    
        X,Y = LinXY(x,y)
        Xerr,Yerr = LinXYerr(xerr,yerr,x,y)
        dx = (max(X)-min(X))/10
        dy = (maxy-min(Y))/20
        xlim = (-0.01,0.45)
        ylim = (-0.001,0.03)
    
    
        popt, pcov = curve_fit(LinRegr1,X,Y,p0=((max(Y)-min(Y))/(max(X)-min(X)),X[0]))
        stdDev=np.sqrt(np.diag(pcov))  #extracting the standard deviation of a and b from pcov (diagonally)
    
        #------setting up the deviating lines (diagonals in 1sigma-range) --------
        t_for_sigma = np.linspace(min(X),max(X),500)
        t1 = np.linspace(xlim[0],xlim[1], 500)
        
        """
        ErrDiag11 = LinRegr((popt[0]+stdDev[0],popt[1]-stdDev[1]),t_for_sigma) #maximal slope a, minimal y intercept b
        ErrDiag21 = LinRegr((popt[0]-stdDev[0],popt[1]+stdDev[1]),t_for_sigma) #minimal slope a, maximal y intercept b
    
        ErrAr_top11 = [t1[-1],ErrDiag11[-1]]
        ErrAr_top21 = [t1[0],ErrDiag21[0]]
        ErrAr_bot11 = [t1[-1],ErrDiag21[-1]]
        ErrAr_bot21 = [t1[0],ErrDiag11[0]]
        ErrAr_top1 = (ErrAr_top11[1]-ErrAr_top21[1])/(ErrAr_top11[0]-ErrAr_top21[0])*(t1-ErrAr_top21[0])+ErrAr_top21[1]
        ErrAr_bottom1 = (ErrAr_bot11[1]-ErrAr_bot21[1])/(ErrAr_bot11[0]-ErrAr_bot21[0])*(t1-ErrAr_bot21[0])+ErrAr_bot21[1]
        centerX, centerY = LineIntersection(popt[0]+stdDev[0],popt[0]-stdDev[0],popt[1]-stdDev[1],popt[1]+stdDev[1])
        """
        
       #------Adding the fitted lines and the measurements to figure and axis--------
        #ax.plot(t1,LinRegr(popt,t1), marker = 'None', linestyle = '-', label=ylabel[:-1]+'=[\eta ] \cdot '+xlabel[1:-1]+'+\eta_0$') #optimized line
        #ax.plot(t_for_sigma,ErrDiag11, marker = 'None', linestyle = 'dashed') #deviating line 1 crossing the 1sigma-range
        #ax.plot(t_for_sigma,ErrDiag21, marker = 'None', linestyle = 'dashed') #deviating line 2 crossing the 1sigma-range
        ax.errorbar(x=X,y=Y,xerr=Xerr,yerr=Yerr,marker='x', color = colorlist[shear_ind],linestyle='None',label=r'Messwerte für $\dot\gamma='+str(round(df.iloc[shear_ind,7],3))+'\mathrm s^{-1}$')
        #ax.fill_between(t_for_sigma,ErrAr_bottom1,ErrAr_top1, alpha = .6, label = r'$1\sigma-\text{Fehlerstreifen}$') #Area containing ~33% of all points of the set (1sigma-range) 
        #ax.scatter(centerX,centerY,marker='v', label = 'Schwerpunkt', color='maroon', edgecolor='black') #geometric center of 1sigma-range
        
        if shear_ind == 0:
            ax.plot(t1,LinRegr(popt,t1), marker = 'None',color=colorlist[shear_ind], linestyle = '-', label=ylabel[:-1]+'=[\eta ] \cdot '+xlabel[1:-1]+'+\eta_0$')
            #ax.fill_between(t_for_sigma,ErrAr_bottom1,ErrAr_top1,color=colorlist[shear_ind], alpha = .3, label = r'$1\sigma-\text{Fehlerstreifen}$') #Area containing ~33% of all points of the set (1sigma-range) 
        else:
           ax.plot(t1,LinRegr(popt,t1), marker = 'None',color=colorlist[shear_ind], linestyle = '-')
           #ax.fill_between(t_for_sigma,ErrAr_bottom1,ErrAr_top1,color=colorlist[shear_ind], alpha = .3) #Area containing ~33% of all points of the set (1sigma-range) 
        
        poptList.append(popt)
        stdList.append(stdDev)
        gammadotList.append(df.iloc[shear_ind,7])
        
    sigfig = 2
    stdDevFlat = np.array(stdList).flatten()*10e3
    poptFlat = np.array(poptList).flatten()*10e3
    stdDev_rounded = ['{:g}'.format(float('{:.{p}g}'.format(stdDevFlat[i], p=sigfig)))+'$' for i in range(0,len(stdDevFlat))]
    decimals = [len(str(stdDev_rounded[i].split('.')[-1])) for i in range(0,len(stdDev_rounded))]
    popt_rounded = [r'$'+str(round(poptFlat[i],decimals[i]))+'\pm\ ' for i in range(len(poptFlat))]
    poptList = np.reshape(np.array(popt_rounded),np.array(poptList).shape)
    stdList = np.reshape(np.array(stdDev_rounded),np.array(stdList).shape)
    cell_text = np.core.defchararray.add(np.transpose(poptList), np.transpose(stdList).astype(str))
    
    
    df = pd.read_csv('sac_40_percent.csv',sep=',',header=None)
    y_excl = df.iloc[:,8]
    x_excl = np.ones(len(y_excl))*.4
    ax.errorbar(x=x_excl,y=y_excl,marker='x', color = 'red',linestyle='None',label='von Fit ausgeschlossen')
    
    the_table = the_table = plt.table(cellText=cell_text,
                    #rowLabels = np.full(7,'a'),
                      rowLabels=[r'$\frac{[\eta ]}{\mathrm{mPa}}$',r'$\frac{\eta_0}{\mathrm{mPa}}$'],
                      colLabels=[r'$\dot\gamma=%.3f\mathrm s^{-1}$'%i for i in gammadotList],#['Wert','Unsicherheit'],
                      loc='bottom',
                      cellLoc='center',
                      bbox=[-0.1,-0.42,1.2,0.3],
                      edges='closed')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)
    
    ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlabel(xlabel,size=16)
    ax.set_ylabel(ylabel,size=16)
    ax.set_xlim(xlim[0],xlim[1])
    ax.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)

    ax.legend(loc='center',ncol=2,fontsize=10,labelspacing=.3)
    fig.tight_layout()

    #----Save Figure-----------
    plt.savefig("Viscosity_by_Concentration_Sucrose_all_Shearrates.pdf",dpi=1200)
        
if __name__ == "__main__":
    plot(['sac_00_percent','sac_10_percent','sac_20_percent'])