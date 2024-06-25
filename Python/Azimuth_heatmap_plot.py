# import necessary libraries for heatmap plotting
import numpy as np
import pandas as pd
import os
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import glob
import scipy.interpolate as spi
from plot import *


DEBUG = True

# Readcsv function that returns the data 
def readCSV(filename):
    data = np.loadtxt(filename, dtype = np.int32, delimiter=' ')
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

#### Main ####

classname = input("Please Input Class Name \n:>")
#dataset_path = os.getcwd() + "/Dataset/"   #dataset path for raspberry
dataset_path = os.getcwd() + "\\Dataset\\" + "\\Azimuth\\" + classname #dataset path for windows
if DEBUG:
    print('dataset path = ',dataset_path)

filenames = os.listdir(dataset_path)
if DEBUG:
    print('filenames = ',filenames)

# use glob to get all the csv files in the folder
csv_files = sorted(glob.glob(os.path.join(dataset_path, '*.csv')))
if DEBUG:
    print('csv_files : ',csv_files)

min_m, max_m = min_max(filenames)

# loop over the list of csv files

# Var initialization
grid_init = 0
colorbar_init = 0
count = 0
range_res = 0.0436                    # Check range_res in config file and adapt before plotting

plt.ion()
for f in csv_files:
    a = readCSV(f)

    df = np.divide(a, max_m - min_m)        # Data normalization

    range_bins = a.shape[0]
    angle_bins = a.shape[1]

    # Plotting grid initialization
    if grid_init == 0:
        theta = np.arcsin(np.linspace(-angle_bins / 2 + 1, angle_bins / 2 - 1, angle_bins) * (2 / angle_bins))  # Angular linear space for plotting
        range = np.linspace(0, range_bins - 1, range_bins) * range_res                                          # Range linear space for plotting
        range = np.maximum(range,0)                                                                                 # Keep only positive range value (later add range bias correction)
        range_depth = range_bins * range_res                                                         
        range_width, grid_res = range_depth/2, 64
        if DEBUG:
            print("max_range:",range_depth)
            print("range bins:",range_bins)
            print("angle_bins:",angle_bins)

        # Grid construction
        posX = np.outer(range, np.sin(theta))
        posY = np.outer(range, np.cos(theta))
        xlin = np.linspace(-np.floor(range_width), np.ceil(range_width), angle_bins)
        ylin = np.linspace(0, range_depth, range_bins)
        xgrid, ygrid = np.meshgrid(xlin, ylin)
        ra_grid = spi.griddata((posX.flatten(), posY.flatten()), df.flatten(),(xgrid, ygrid), method='cubic')
        grid_init = 1
    
    hmplot = plt.contourf(xlin,ylin,df,cmap='Spectral_r')
    #hmplot.axes.set_ylim(0,10)
    #hmplot.axes.set_xlim(-5,5)

    if colorbar_init == 0:
        plt.colorbar(hmplot)
        colorbar_init = 1
    plt.title(csv_files[count])

    plt.show()
    plt.pause(0.25)
    count+=1

plt.ioff()
