# Import necessary libraries for heatmap plotting
import numpy as np
import pandas as pd 
import os
from os.path import join
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import glob
import scipy.interpolate as spi

# Configuration file name
configFileName = os.getcwd() + '\\config_files\\test.cfg'

DEBUG = False

# Readcsv function that returns the data 
def readCSV(filename):
    data = np.loadtxt(filename, dtype=np.int32, delimiter=' ')
    return data

# MIN-MAX Function
def min_max(in_files):
    min_val = 5000
    max_val = 0
    
    for index, filename in enumerate(in_files):
        path = join(dataset_path, filename)
        if DEBUG:
            print('target = ', path)
        if not path.endswith('.csv'):
            continue
        heatmap = readCSV(path)
        if DEBUG:
            print(heatmap.shape)
        min_val = np.minimum(min_val, np.min(heatmap))
        max_val = np.maximum(max_val, np.max(heatmap))
        if DEBUG:
            print("MIN: " + str(min_val) + "\tMAX: " + str(max_val))
            print(index)

    print("MIN: " + str(min_val) + "\tMAX: " + str(max_val))
    return min_val, max_val

# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    configParameters = {}
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        splitWords = i.split(" ")

        global numRxAnt, numTxAnt
        numRxAnt = 4
        numTxAnt = 2

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

        elif "frameCfg" in splitWords[0]:
            chirpStartIdx = int(splitWords[1])
            chirpEndIdx = int(splitWords[2])
            numLoops = int(splitWords[3])
            numFrames = int(splitWords[4])
            framePeriodicity = int(splitWords[5])

    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)

    if DEBUG:
        print(configParameters)

    return configParameters

#### Main ####
classname = input("Please Input Class Name \n:>")
dataset_path = os.getcwd() + "\\Dataset\\" + "Azimuth\\" + classname

if DEBUG:
    print('dataset path = ', dataset_path)

filenames = os.listdir(dataset_path)
if DEBUG:
    print('filenames = ', filenames)

csv_files = sorted(glob.glob(os.path.join(dataset_path, '*.csv')))
if DEBUG:
    print('csv_files : ', csv_files)

min_m, max_m = min_max(filenames)

grid_init = 0
cb_init = 0
count = 0

configParameters = parseConfigFile(configFileName)
range_res = configParameters["rangeResolutionMeters"]

plt.ion()
for f in csv_files:
    a = readCSV(f)
    a = (a - min_m) / (max_m - min_m)

    range_bins = a.shape[0]
    angle_bins = a.shape[1]

    if grid_init == 0:
        theta = np.arcsin(np.linspace(-angle_bins / 2 + 1, angle_bins / 2 - 1, angle_bins) * (2 / angle_bins))
        r = np.array(range(range_bins)) * range_res

        range_depth = configParameters["numRangeBins"] * range_res
        range_width, grid_res = range_depth/2, 400

        posX = np.outer(r.T, np.sin(theta)).flatten()
        posY = np.outer(r.T, np.cos(theta)).flatten()
        xlin = np.linspace(-int(range_width), int(range_width), grid_res)
        ylin = np.linspace(0, range_depth, grid_res)

        xiyi = np.meshgrid(xlin, ylin)
        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes()
        ax.imshow(((0,)*grid_res,) * grid_res, cmap=plt.cm.jet, extent=[-range_width, +range_width, 0, range_depth], alpha=0.95)
        ax.set_xlabel('Lateral distance along [m]')
        ax.set_ylabel('Longitudinal distance along [m]')

        grid_init = 1

    if DEBUG:
        print("posX shape:", posX.shape)
        print("posY shape:", posY.shape)
        print("a shape:", a.flatten().shape)
        print("xiyi[0] shape:", xiyi[0].shape)
        print("xiyi[1] shape:", xiyi[1].shape)
    
    zi = spi.griddata((posX, posY), a.flatten(), (xiyi[0], xiyi[1]), method='linear')
    zi = zi.reshape(len(ylin), len(xlin))
    
    plt.contourf(xlin, ylin, zi, cmap='jet')
    plt.title(f)
    plt.show()
    plt.pause(0.5)
    count += 1

plt.ioff()
