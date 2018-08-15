#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 20:02:32 2017

@author: jacoblashner
"""

import Telescope as tp
import thermo as th
import numpy as np
import matplotlib.pyplot as plt
import json
from StringIO import StringIO

pW = 10 ** 12
GHz = 10**9

config = json.load(open("../run/config.json"))
config["theta"] = np.deg2rad(20.)
config["WriteOutput"] = False


config["bandID"] = 2
config["ExperimentDirectory"] = "../Experiments/V3/MF_baseline/SmallTelescope"
config["HWPDirectory"] = "../HWP/4LayerSapphire"    
config["Reflection Order"] = 10
tel = Telescope(config)

print("\n HWPSS output\n" + "-"* 20)
print("A2: %f K_cmb"%(tel.A2_Kcmb))
print("a2: %f %%"%(tel.a2 * 100))
print("A4: %f K_cmb"%(tel.A4_Kcmb))
print("a4: %f %%"%(tel.a4 * 100))
print("-"*20)