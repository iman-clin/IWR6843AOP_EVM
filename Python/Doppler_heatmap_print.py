import csv
from ast import literal_eval
from pprint import pprint
from os import listdir
from os.path import isdir, join
import numpy as np
import os
from numpy import loadtxt
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
# import necessary libraries
import pandas as pd
import os
import glob
import time

DEBUG = True

def readCSV(filename):
    data = loadtxt(filename, dtype = np.float32, delimiter=' ')
    return data

classname = input("Please Input Class Name \n:>")
#dataset_path = os.getcwd() + "/Dataset/"   #dataset path for raspberry
dataset_path = os.getcwd() + "\\Dataset\\" + classname  #dataset path for windows
if DEBUG:
    print('dataset path = ',dataset_path)

filenames = listdir(dataset_path)
if DEBUG:
    print('filenames = ',filenames)

# MIN-MAX Function
def min_max(in_files):
    # Numpy arrays to store train, test and val matrix
    min = 5000
    max = 0
    count = 0
    
    for index, filenames in enumerate(in_files):
        
        # Create path from given filename and target item
        path = join(dataset_path,filenames)
        if DEBUG:
            print('target = ',path)
        
        # Check to make sure we're reading a .wav file
        if not path.endswith('.csv'):
            continue
        

        heatmap = readCSV(path)
        if DEBUG:
            print(heatmap.shape)
        
        min = np.minimum(min, np.min(heatmap))
        max = np.maximum(max, np.max(heatmap))
        if DEBUG:
            print("MIN: " + str(min) + "\tMAX: " + str(max))
            print(index)
        count += 1
        
    print("Count: " + str(count))
    print("MIN: " + str(min) + "\tMAX: " + str(max))
            
    return min, max
 

global min_m
global max_m
min_m, max_m = min_max(filenames)

# use glob to get all the csv files 
# in the folder


csv_files = sorted(glob.glob(os.path.join(dataset_path, '*.csv')))
if DEBUG:
    print('csv_files : ',csv_files)

csv_reduzido = sorted(glob.glob('*.csv'))
if DEBUG:
    print('csv_reduced : ',csv_reduzido)

# loop over the list of csv files
count = 0
for f in csv_files[count:count+50]:
    # read the csv file
    heatmap = readCSV(f)
    aux_n1 = np.subtract(heatmap, min_m)
    df = np.divide(aux_n1, max_m - min_m)
    df[0,0] = 0
    df[0,1] = 1
    plt.imshow(df, cmap='Spectral_r', interpolation='nearest', aspect='auto')
    plt.title(csv_files[count]+'_new')
    plt.show()
    energia_calculada = np.sum(np.power(df, 2))
    print(energia_calculada)  
    count+=1
print(csv_files)
