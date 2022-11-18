# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:52:09 2022

@author: Jan-Philipp
"""

import numpy as np

x,dx,U,dU = np.loadtxt('Col_z_5.csv',delimiter=';',skiprows=1,unpack=True)
print(type(x))