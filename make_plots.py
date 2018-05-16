#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 08:48:40 2018

@author: jacoblashner
"""

import numpy as np
import matplotlib.pyplot as plt

theta, A2, A4, a2, a4 = np.loadtxt("HWPSS_vs_theta.txt", 
                                   skiprows=1, unpack = True)

plt.plot(theta, A2 * 1000)
plt.plot(theta, A4 * 1000)
plt.xlabel("Incident Angle (deg)", fontsize = 15)
plt.ylabel('HWPSS (mK$_{\\rm CMB}$)', fontsize = 15)
plt.legend(["A2", "A4"], fontsize = 15)
plt.savefig("/Users/jacoblashner/Desktop/HWPSS.pdf")
plt.show()
