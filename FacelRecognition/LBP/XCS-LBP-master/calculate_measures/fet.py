# -*- coding: utf-8 -*-
"""
FET - FOREGROUND EVALUATION TOOL

@author: Andrews Sobral
"""

import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join

path_gt = 'GT/'
path_fg = 'FG/'
path_sc = 'SC/'

if not os.path.exists(path_sc):
    os.makedirs(path_sc)


included_extensions = ['jpg', 'bmp', 'png', 'gif']
files_gt = [ f for f in listdir(path_gt) if any(f.endswith(ext) for ext in included_extensions) ]
files_fg = [ f for f in listdir(path_gt) if any(f.endswith(ext) for ext in included_extensions) ]
# files_gt = [ f for f in listdir(path_gt) if isfile(join(path_gt,f)) ]
# files_fg = [ f for f in listdir(path_gt) if isfile(join(path_gt,f)) ]

TP = .0 # True positive pixels
FP = .0 # False positive pixels
TN = .0 # True negative pixels
FN = .0 # False negative pixels
Recall = .0 # TP / (TP + FN)
Precision = .0 # TP / (TP + FP)
Fscore = .0 # 2*(Precision*Recall)/(Precision+Recall)

# TP - # of foreground pixels classified as foreground pixels.
# FP - # of background pixels classified as foreground pixels.
# TN - # of background pixels classified as background pixels.
# FN - # of foreground pixels classified as background pixels.

green = [0,255,0] # for FN
blue  = [255,0,0] 
red   = [0,0,255] # for FP
white = [255,255,255] # for TP
black = [0,0,0] # for TN

print ('Processing')
k = 1
for file_gt, file_fg in zip(files_gt, files_fg):
    print(k, file_gt, file_fg)
    img_gt = cv2.imread(path_gt + file_gt,cv2.IMREAD_GRAYSCALE)
    img_gt = cv2.resize(img_gt, (0,0), fx=0.5, fy=0.5)
    img_fg = cv2.imread(path_fg + file_fg,cv2.IMREAD_GRAYSCALE)
    #print(img_gt.shape,img_fg.shape)
    rows,cols = img_gt.shape
    img_fg = cv2.resize(img_fg, (cols, rows))
    #print(img_gt.shape,img_fg.shape)
    img_res = np.zeros((rows,cols,3),np.uint8)
    for i in range(rows):
        for j in range(cols):
            pixel_gt = img_gt[i,j]
            pixel_fg = img_fg[i,j]        
            if(pixel_gt == 255 and pixel_fg == 255):
                TP = TP + 1
                img_res[i,j] = white
            if(pixel_gt == 0 and pixel_fg == 255):
                FP = FP + 1
                img_res[i,j] = red
            if(pixel_gt == 0 and pixel_fg == 0):
                TN = TN + 1
                img_res[i,j] = black
            if(pixel_gt == 255 and pixel_fg == 0):
                FN = FN + 1
                img_res[i,j] = green
    cv2.imshow('GT',img_gt)
    cv2.imshow('FG',img_fg)
    cv2.imshow('SC',img_res)
    cv2.imwrite(path_sc + file_gt, img_res)
    cv2.waitKey(1) # 33
    k = k + 1
    #break
cv2.destroyAllWindows()

Recall = TP / (TP + FN)
Precision = TP / (TP + FP)
Fscore = 2*Precision*Recall/(Precision+Recall)

print ('Score:')
print ('TP: ', TP)
print ('FP: ', FP)
print ('TN: ', TN)
print ('FN: ', FN)
print ('Recall: ', Recall)
print ('Precision: ', Precision)
print ('Fscore: ', Fscore)
print ('')

with open("score.txt", "w") as text_file:
	text_file.write("Score:\n")
	text_file.write("TP: %d\n" % TP)
	text_file.write("FP: %d\n" % FP)
	text_file.write("TN: %d\n" % TN)
	text_file.write("FN: %d\n" % FN)
	text_file.write("Recall:     %f\n" % Recall)
	text_file.write("Precision:  %f\n" % Precision)
	text_file.write("Fscore:     %f\n" % Fscore)
	text_file.write("\n")

#####################################################################

