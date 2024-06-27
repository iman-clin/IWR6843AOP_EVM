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

# Configuration file name
configFileName = os.getcwd() + '\\config_files\\test.cfg'

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

# ------------------------------------------------------------------

# Function to parse the data inside the configuration file

def parseConfigFile(configFileName):
    global RANGE_FFT_SIZE, DOPPLER_FFT_SIZE
    configParameters = {} # Initialize an empty dictionary to store the configuration parameters

    # Read the configuration file to extract config parameters and frame config
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:

        # Split the line
        splitWords = i.split(" ")

        # Hard code the number of antennas, change if other configuration is used
        global numRxAnt, numTxAnt
        numRxAnt = 4
        numTxAnt = 2

        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1
            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 *= 2
            digOutSampleRate = int(splitWords[11])

        # Get the information about the frame configuration 
        elif "frameCfg" in splitWords[0]:
            chirpStartIdx = int(splitWords[1])
            chirpEndIdx = int(splitWords[2])
            numLoops = int(splitWords[3])
            numFrames = int(splitWords[4])
            framePeriodicity = int(splitWords[5])

    # Combine the read data to obtain the configuration parameters 
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
    RANGE_FFT_SIZE = int(configParameters["numRangeBins"])
    DOPPLER_FFT_SIZE = int(configParameters["numDopplerBins"] - 1)
    if DEBUG:
        print(configParameters)

    return configParameters




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

# Gather utile infos from config file
configParameters = parseConfigFile(configFileName)
range_res = configParameters["rangeResolutionMeters"]
range_depth = configParameters["maxRange"]
range_width = range_depth/2

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
        grid_res = 3000

        # Grid construction
        posX = np.outer(range, np.sin(theta))
        posY = np.outer(range, np.cos(theta))
        xlin = np.linspace(-np.floor(range_width), np.ceil(range_width), angle_bins)
        ylin = np.linspace(0, range_depth, range_bins)
        xgrid, ygrid = np.meshgrid(xlin, ylin)
        ra_grid = spi.griddata((posX.flatten(), posY.flatten()), df.flatten(),(xgrid, ygrid), method='cubic')
        grid_init = 1
    
    hmplot = plt.contourf(xlin,ylin,df,cmap='Spectral_r')
    hmplot.axes.set_ylim(0,range_depth)
    #hmplot.axes.set_xlim(-5,5)

    if colorbar_init == 0:
        plt.colorbar(hmplot)
        colorbar_init = 1
    plt.title(csv_files[count])

    plt.show()
    plt.pause(0.25)
    count+=1

plt.ioff()
