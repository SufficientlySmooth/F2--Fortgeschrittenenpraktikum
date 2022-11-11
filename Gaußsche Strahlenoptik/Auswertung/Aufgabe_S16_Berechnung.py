# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 17:33:46 2022

@author: Jan-Philipp
"""
from numpy import sqrt, pi
lambd = 632.9 * 10**-9
R = 50 * 10**-3
b = 6.35 * 10**-3
d = 45 * 10**-3
z_prime = - d/2
n = 1.515
w0_prime = sqrt(lambd/pi*sqrt(d/2*(R-d/2)))

A = b*(1-n)/R+1
B = b*n
C = (1-n)/(n*R)
D = 1

zR_prime = pi*w0_prime**2/lambd

z = (A*D*z_prime+B*C*D*z_prime-A*B*D-C*D*z_prime**2-C*D*zR_prime**2)/((A-C*z_prime)**2+C**2*zR_prime**2)

zR = (A*D*zR_prime-B*C*D*zR_prime)/((A-C*z_prime)**2+C**2*zR_prime**2)