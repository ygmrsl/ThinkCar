# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 00:20:11 2021

@author: ygmr
"""

import os
from glob import glob
import pandas as pd
import numpy as np
#%%
inputdir = r"E:\carla\CARLA_0.9.10\WindowsNoEditor\PythonAPI\examples\_out"
pths = sorted(glob(inputdir+"/*.png"))
#%%
filtered = []

for pth in pths:
    if "combined" not in pth and "mask" not in pth:
        filtered.append(pth)
#%%

images = np.repeat(filtered, 5)
speed = np.tile([0,1,2,3,4], len(filtered))
label = np.zeros(len(speed))
data = {"pth":images, "speed":speed, "label":label}

df = pd.DataFrame(data, columns=list(data.keys()))

df.to_csv(r"E:\GitHub\ThinkCar/ethics.csv", index=False)
