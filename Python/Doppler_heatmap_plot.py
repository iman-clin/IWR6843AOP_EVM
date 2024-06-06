# import necessary libraries for heatmap plotting
import numpy as np
import pandas as pd
import os
from os.path import join
import matplotlib.pyplot as plt
import glob

DEBUG = False

# Readcsv function that returns the data 
def readCSV(filename):
    data = np.loadtxt(filename, dtype = np.float32, delimiter=' ')
    return data

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

##################### Loop for heatmap printing #####################

classname = input("Please Input Class Name \n:>")
#dataset_path = os.getcwd() + "/Dataset/"   #dataset path for raspberry
dataset_path = os.getcwd() + "\\Dataset\\" + classname  #dataset path for windows
if DEBUG:
    print('dataset path = ',dataset_path)

filenames = os.listdir(dataset_path)
if DEBUG:
    print('filenames = ',filenames)

global min_m
global max_m
min_m, max_m = min_max(filenames)

# use glob to get all the csv files in the folder
csv_files = sorted(glob.glob(os.path.join(dataset_path, '*.csv')))
if DEBUG:
    print('csv_files : ',csv_files)

 
# loop over the list of csv files
count = 0
plt.ion()
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
    plt.pause(0.25)
    energia_calculada = np.sum(np.power(df, 2))     # Calculate energy deployed on each frame
    print(energia_calculada)  
    count+=1

