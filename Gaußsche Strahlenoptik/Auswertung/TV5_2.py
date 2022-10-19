# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 10:23:36 2022

@author: Jan-Philipp Christ
"""
import numpy as np #math functions, arrays
import matplotlib.pyplot as plt #visualizing
from scipy.optimize import curve_fit
import matplotlib

matplotlib.style.use('JaPh') #merely contains basic style information
plt.ioff()

def LinXY(x,y):
    """"Scale x and y to obtain an linear relation between ordinate and abscissa"""
    return x,y

def LinXYerr(xerr,yerr,x,y):
    """"Scale xerr and yerr according to LinXY"""
    return xerr,yerr

def LinRegr(tup,x):
    """Model of a linear function for fitting and plotting"""
    a,b = tup
    return a*x+b

def LinRegrCurveFit(x,a,b):
    return a*x+b

def LineIntersection(m1,m2,b1,b2):
    """determining the geometric center of the 1-sigma-range"""  
    x = (b2-b1)/(m1-m2)
    y = m1*x+b1
    return x,y

def plot(PDFname):
    global x,y
    fig,ax=plt.subplots(1,1,figsize=(10,10/np.sqrt(2))) #create axis embedded in figure

    x = np.array(list(range(35,51)))/10

    y = np.array([23.6,23.0,22.2,20.0,20.4,19.4,18.4,17.6,16.6,15.6,14.6,13.6,12.4,11.2,10.0,8.2])

    xerr_abs = 0.01
    yerr_abs = 0.2
    xerr_rel = 0
    yerr_rel = 0
    
    xerr = x*xerr_rel+xerr_abs
    yerr = y*yerr_rel+yerr_abs
    
    ylabel = r'$\dfrac{\mathrm{f}}{\mathrm{kHz}}$' 
    xlabel = r'$\dfrac{\mathrm{y}}{\mathrm{cm}}$'
    
    X,Y = LinXY(x,y)
    Xerr,Yerr = LinXYerr(xerr,yerr,x,y)
    
    xlim = (3.4,5.1)
    ylim = (7,25)
    
    popt,pcov = curve_fit(LinRegrCurveFit,x,y)
    stdDev=np.sqrt(np.diag(pcov))  #extracting the standard deviation of a and b from pcov (diagonally)
    
    #------setting up the deviating lines (diagonals in 1sigma-range) --------
    t1 = np.linspace(xlim[0],xlim[1], 10**4)
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
    ax.plot(t1,LinRegr(popt,t1), marker = 'None', linestyle = '-')#, label=ylabel[:-1]+'_{opt.}=%.6f\cdot'%popt[0]+xlabel[1:-1]+'%+.6f$'%popt[1]) #optimized line
    ax.plot(t1,ErrDiag11, marker = 'None', linestyle = 'dashed')#, label=ylabel[:-1]+'_{Err1}=%.5f\cdot'%(popt[0]+stdDev[0])+xlabel[1:-1]+'%+.5f$'%(popt[1]-stdDev[1])) #deviating line 1 crossing the 1sigma-range
    ax.plot(t1,ErrDiag21, marker = 'None', linestyle = 'dashed')#, label=ylabel[:-1]+'_{Err2}=%.5f\cdot'%(popt[0]-stdDev[0])+xlabel[1:-1]+'%+.5f$'%(popt[1]+stdDev[1])) #deviating line 2 crossing the 1sigma-range
    ax.errorbar(x=X,y=Y,xerr=Xerr,yerr=Yerr,marker='x',linestyle='None')#,label='Messwerte')
    ax.fill_between(t1,ErrAr_bottom1,ErrAr_top1, alpha = .6)#, label = r'$1\sigma-\text{Fehlerstreifen}$') #Area containing ~33% of all points of the set (1sigma-range) 
    ax.scatter(centerX,centerY,marker='v', color='maroon', edgecolor='black') #geometric center of 1sigma-range
    
    
    left19 = (19.5-ErrAr_bot21[1])/((ErrAr_bot11[1]-ErrAr_bot21[1])/(ErrAr_bot11[0]-ErrAr_bot21[0])) + ErrAr_bot21[0]
    right19 = (19.5-ErrAr_top21[1])/((ErrAr_top11[1]-ErrAr_top21[1])/(ErrAr_top11[0]-ErrAr_top21[0])) + ErrAr_top21[0]
    left23 =  (23.5-ErrAr_bot21[1])/((ErrAr_bot11[1]-ErrAr_bot21[1])/(ErrAr_bot11[0]-ErrAr_bot21[0])) + ErrAr_bot21[0]
    right23 = (23.5-ErrAr_top21[1])/((ErrAr_top11[1]-ErrAr_top21[1])/(ErrAr_top11[0]-ErrAr_top21[0])) + ErrAr_top21[0]
    
    ax.axhline(23.5,marker='None',linestyle='-',color='darkseagreen')
    ax.axhline(19.5,marker='None',linestyle='-',color='navajowhite')
    
    ax.axvline((23.5-popt[1])/popt[0],marker='None',linestyle='dotted',color='darkseagreen', label = r'$y_{@ 23.5\mathrm{kHz}}=(%.3f\pm %.3f)\ \mathrm{cm}$'%((23.5-popt[1])/popt[0],abs((23.5-popt[1])/popt[0]-left23)))
    ax.axvline((19.5-popt[1])/popt[0],marker='None',linestyle='dotted',color='navajowhite',label = r'$y_{@ 19.5\mathrm{kHz}}=(%.3f\pm %.3f)\ \mathrm{cm}$'%((19.5-popt[1])/popt[0],abs((19.5-popt[1])/popt[0]-left19)))
    
    ax.axvspan(left19,right19,color='navajowhite',alpha=.5,linestyle='None')
    ax.axvspan(left23,right23,color='darkseagreen',alpha=.5,linestyle='None')
    
    #-----Formatting------------
    ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlabel(xlabel,size=16)
    ax.set_ylabel(ylabel,size=16)
    ax.set_xlim(xlim[0],xlim[1])
    ax.grid(visible=True, which='minor', color="grey", linestyle='-',linewidth=.008, alpha=.2)
    ax.legend(loc='upper center',fontsize=15,bbox_to_anchor=(0.5, -0.13),ncol=2)
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),ncol=2,fontsize=15)
    #ax.set_title('Messreihe xy',pad=10,fontsize=16)

    fig.tight_layout()

    #----Save Figure-----------
    plt.savefig(PDFname+".pdf",dpi=1200)
        
if __name__ == "__main__":
    plot('TV5_plot2')