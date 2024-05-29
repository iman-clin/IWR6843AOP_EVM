# import necessary libraries for heatmap plotting
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
import os
import glob

# import necessary libraries for CNN building and data processing
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler

DEBUG = False

# CNN class definition 

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(                                # Layers 1 to n
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU())
        self.fc = nn.Linear(7*7*32, num_classes)                    # Convolution filter after data has been processed

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
    

##################### Functions #####################

# Readcsv function that returns the data 
def readCSV(filename):
    data = loadtxt(filename, dtype = np.float32, delimiter=' ')
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
 
def readCSVforCNN(filename):
    data = pd.read_csv(filename)
    return data



##################### Loop for heatmap printing #####################

classname = input("Please Input Class Name \n:>")
#dataset_path = os.getcwd() + "/Dataset/"   #dataset path for raspberry
dataset_path = os.getcwd() + "\\Dataset\\" + classname  #dataset path for windows
if DEBUG:
    print('dataset path = ',dataset_path)

filenames = listdir(dataset_path)
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
    energia_calculada = np.sum(np.power(df, 2))     # Calculate energy deployed on each frame
    print(energia_calculada)  
    count+=1



##################### CNN usage #####################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     # On GPU if cuda is avalaible
model = CNN(10).to(device)

for f in csv_files:
    data = pd.read_csv(f)
    features = data.iloc[:,:-1].values
    labels = data.iloc[:,-1].values