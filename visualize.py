# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 00:10:29 2021

@author: admin
"""


import os
from glob import glob
import cv2
import matplotlib.pyplot as plt
from PIL import Image

#%%

inputdir = r"E:\carla\CARLA_0.9.10\WindowsNoEditor\PythonAPI\examples\_out"
pths = sorted(glob(inputdir+"/*.png"))

#%%
for pth in pths:
    if "mask" in pth or "combined" in pth:
        continue
    base = os.path.splitext(pth)[0]
    mask = base + "-mask.png"
    img = base + ".png"
    
    image = Image.open(img).convert("RGB")
    pred = Image.open(mask)
    
    combined = Image.blend(image, pred, 0.4)
    
    combined.save(base+"-combined.png")
#%%
pths = sorted(glob(inputdir+"/*.png"))
image_array = []
size = None
for pth in pths:
    if "combined" in pth:
        image = cv2.imread(pth)
        #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_array.append(image)
        h, w, l = image.shape
        size = (w,h)

vid = cv2.VideoWriter(r"E:\videolar\test_long3.avi", cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
for i in image_array:
    vid.write(i)
vid.release()

#cv2.VideoWriter_fourcc(*'DIVX')
