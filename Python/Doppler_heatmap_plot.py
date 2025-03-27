# import necessary libraries for heatmap plotting
import numpy as np
import pandas as pd
import os
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import glob

DEBUG = True

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

##################### Heatmap printing #####################

classname = input("Please Input Class Name \n:>")    
dataset_path = os.getcwd() + "/MTZ_experiment/dataset/" + classname            #dataset path for windows

filenames = os.listdir(dataset_path)

# use glob to get all the csv files in the folder
csv_files = sorted(glob.glob(os.path.join(dataset_path, '*.csv')))

if DEBUG:
    print('dataset path = ',dataset_path)
    print('filenames = ',filenames)
    print('csv_files : ',csv_files)

global min_m
global max_m
min_m, max_m = min_max(filenames)

# Plot construction and update
fig, axes = plt.subplots()

def update(count):
    f = csv_files[count]                    # Load and read csv file
    heatmap = readCSV(f)
    aux_n1 = np.subtract(heatmap, min_m)    # Data formatting
    df = np.divide(aux_n1, max_m - min_m)
    df[0,0] = 0
    df[0,1] = 1
    axes.clear()
    axes.imshow(heatmap, cmap='Spectral_r', interpolation='nearest', aspect='auto')
    plt.title(filenames[count])
    return 0


ani = anim.FuncAnimation(fig = fig, func = update, frames = len(csv_files), interval = 250)
plt.show()
