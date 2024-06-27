import os
import sys
import serial
import time
import numpy as np
import pandas as pd
import tkinter as tk
from PIL import ImageTk, Image
from keras import models

# Configuration file name
configFileName = os.getcwd() + '\\config_file_points.cfg'

# CNN model, min and max values and threshold
model_name = os.getcwd() + '\\all_targets_1605_5349.h5'
min_m = 1605.0
max_m = 5349.0
THRESHOLD = 0.95

# Number of rows and columns for heatmap sample
NUMBER_OF_ROWS = 63
NUMBER_OF_COLUMNS = 128
DEPTH = 4

# Buffer and useful vars
CLIport = {}
Dataport = {}

maxBufferSize = 2**15
byteBuffer = np.zeros(maxBufferSize, dtype='uint8')
byteBufferLength = 0
init = 0

compteur = 0

magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

ObjectsData = 0

DEBUG = False

# ------------------------------------------------------------------

# Function to configure the serial ports and send the data from the configuration file to the radar

def serialConfig(configFileName):
    global CLIport, Dataport

    # Open the serial ports for the configuration and the data ports
    # Raspberry pi   /    Windows 
    try:
        # next 2 lines for connection on Raspberry
        #CLIport = serial.Serial('/dev/ttyUSB0', 115200)   
        #Dataport = serial.Serial('/dev/ttyUSB1', 921600)

        # next 2 lines for connection on Windows
        CLIport = serial.Serial(input("mmWave Demo input config port (enhanced port) = "), 115200)
        Dataport = serial.Serial(input("mmWave Demo input data port = "), 921600)

        # Read the configuration file and send it to the board
        config = [line.rstrip('\r\n') for line in open(configFileName)]
        for i in config:
            CLIport.write((i + '\n').encode())
            if DEBUG:
                print(i)
            time.sleep(0.01)
    
    # Exception on serial ports opening
    except serial.SerialException as se:
        print('Problem Opening Serial Port!! \n Error: ')
        print(str(se) + '\n')
        sys.exit()
    return CLIport, Dataport


# ------------------------------------------------------------------

# Function to parse the data inside the configuration file

def parseConfigFile(configFileName):
    global NUMBER_OF_COLUMNS, NUMBER_OF_ROWS
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
            global numAdcSamples
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
    global numChirpsPerFrame
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
    NUMBER_OF_COLUMNS = int(configParameters["numRangeBins"])
    NUMBER_OF_ROWS = int(configParameters["numDopplerBins"] - 1)

    return configParameters


# ------------------------------------------------------------------

# Function to normalize a matrix

def norm(mat):
    aux_n1 = np.subtract(mat, min_m)
    aux_n2 = np.divide(aux_n1, (max_m - min_m))
    norm_mat = aux_n2.reshape(NUMBER_OF_ROWS,
                              NUMBER_OF_COLUMNS)
    return norm_mat
    
# ------------------------------------------------------------------

#Function to read a full data Packet, from frame header to last data

def readData(Dataport):
    global byteBuffer, byteBufferLength
    byteBuffer = np.zeros(maxBufferSize, dtype='uint8')
    while True:
        readBuffer = Dataport.read(Dataport.in_waiting)
        byteVec = np.frombuffer(readBuffer, dtype='uint8')
        byteCount = len(byteVec)

        # Check that the buffer is not full, and then add the data to the buffer
        if (byteBufferLength + byteCount) < maxBufferSize:
            byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
            byteBufferLength += byteCount

        # Check that the buffer has sufficient amount of data
        if byteBufferLength > 2**14:

             # Check for all possible locations of the magic word
            possibleLocs = np.where(byteBuffer == magicWord[0])[0]

            # Confirm that is the beginning of the magic word and store the index in startIdx
            startIdx = []
            for loc in possibleLocs:
                check = byteBuffer[loc:loc + 8]
                if np.all(check == magicWord):
                    startIdx.append(loc)

            # Check that startIdx is not empty
            if startIdx:

                # Remove the data before the first start index
                if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                    byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                    byteBuffer[byteBufferLength - startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength - startIdx[0]:]), dtype='uint8')
                    byteBufferLength = byteBufferLength - startIdx[0]

                # Check that there are no errors with the byte buffer length  
                if byteBufferLength < 0:
                    byteBufferLength = 0
                
                word = [1, 2**8, 2**16, 2**24]                              # word array to convert 4 bytes to a 32 bit number
                totalPacketLen = np.matmul(byteBuffer[12:12 + 4], word)     # Read the total packet length
                
                # Check that the whole packet has been read
                if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                    break
    return byteBuffer

# ------------------------------------------------------------------

#Function to parse Datas from the read packet

def parseData68xx(byteBuffer):
    global NUMBER_OF_COLUMNS, NUMBER_OF_ROWS
    global res, mat, inpt
    global init, compteur

    dataOK = 0
    word = [1, 2**8, 2**16, 2**24]                                          # word array to convert 4 bytes to a 32 bit number
    idX = 0                                                                 # Initialize the pointer index
    magicNumber = byteBuffer[idX:idX + 8]                                   # Read the header
    idX += 8
    version = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
    idX += 4
    totalPacketLen = np.matmul(byteBuffer[idX:idX + 4], word)
    idX += 4
    platform = format(np.matmul(byteBuffer[idX:idX + 4], word), 'x')
    idX += 4
    frameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
    idX += 4
    timeCpuCycles = np.matmul(byteBuffer[idX:idX + 4], word)
    idX += 4
    numDetectedObj = np.matmul(byteBuffer[idX:idX + 4], word)
    idX += 4
    numTLVs = np.matmul(byteBuffer[idX:idX + 4], word)
    idX += 4
    subFrameNumber = np.matmul(byteBuffer[idX:idX + 4], word)
    idX += 4

    print("Sample #",compteur)
    if DEBUG:
        print('numTLVs :',numTLVs)
    # Read the TLV messages
    for tlvIdx in range(numTLVs):
        # Check the header of the TLV message to find type and length of it
        tlv_type = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        if DEBUG:
            print('\n tlv type :', tlv_type)
        tlv_length = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        if DEBUG:
            print('\n tlv length :', tlv_length)

        # Read the data if TLV type 1 (Detected points) detected
        if tlv_type == 1:
            num_points = tlv_length/16
            if DEBUG:
                print(num_points,"points detected")
            vect = byteBuffer[idX:idX + tlv_length].view(np.uint32)     # Data vector
            points_array = np.zeros([int(num_points),4],dtype='uint32')
            points_array = vect.reshape(int(num_points),4)
            if DEBUG:
                labels = ['X[m]','Y[m]','Z[m]','Doppler[m/s]']
                points_df = pd.DataFrame(points_array,columns=labels)
                print(points_df)


        # Read the data if TLV type 5 (Doppler heatmap) detected
        if tlv_type == 5:
            if DEBUG:
                print(compteur)
            compteur += 1
            resultSize = NUMBER_OF_COLUMNS * (NUMBER_OF_ROWS + 1) * np.dtype(np.uint16).itemsize
            if tlv_length == resultSize:
                if DEBUG:
                    print("Sizes Matches: ", resultSize)
                    print("\nRange Bins: ", NUMBER_OF_COLUMNS)
                    print("\nDoppler Bins: ", NUMBER_OF_ROWS + 1)
                
                ares = byteBuffer[idX:idX + resultSize].view(np.uint16) # Data vector
                res = np.reshape(ares, res.shape)                       # Data array of the right size
                # Shift the data to the correct position
                rest = np.fft.fftshift(res, axes=(1,))      # put left to center, put center to right
                # Transpose the input data for better visualization
                result = np.transpose(rest)
                # Normalize the data
                mat = norm(result[1:])
                if mat.shape == (NUMBER_OF_ROWS, NUMBER_OF_COLUMNS):
                    if init < 4:
                        inpt[:, : , : , init, 0] = mat
                        init += 1
                        dataOK = 0
                    else:
                        dataOK = 1
                        inpt[:, : , : , 0, 0] = inpt[:, : , : , 1, 0]
                        inpt[:, : , : , 1, 0] = inpt[:, : , : , 2, 0]
                        inpt[:, : , : , 2, 0] = inpt[:, : , : , 3, 0]
                        inpt[:, : , : , 3, 0] = mat

                # Remove DC value from matrix
                mat = result[1:, :]
                if mat.shape == (NUMBER_OF_ROWS, NUMBER_OF_COLUMNS):
                    dataOK = 1
                    print(mat)
                else:
                    dataOK = 0
                    print("Invalid Matrix")
                    return dataOK
                break
        
        # Read the data if TLV type 7 (Side info on Detected points) detected
        if tlv_type == 7:
            num_points = tlv_length/4
            if DEBUG:
                print(num_points,"points detected")
            vect_pi = byteBuffer[idX:idX + tlv_length].view(np.uint16)     # Data vector
            pointsinfo_array = np.zeros([int(num_points),2],dtype='uint16')
            pointsinfo_array = vect_pi.reshape(int(num_points),2)
            points_array = np.concatenate((points_array,pointsinfo_array), axis=1)
            labels = ['X[m]','Y[m]','Z[m]','Doppler[m/s]','SNR[dB]','noise[dB]']
            points_df = pd.DataFrame(points_array,columns=labels)
            print(points_df,'\n')

        idX += tlv_length   # Check next TLV
    return dataOK           # Later return points_array too, find a way to add the infos to CNN data


# ------------------------------------------------------------------

label = ['Idle', 'Presence']    # Labels

# Funtion to update the data and display in the plot

def update():
    global model, THRESHOLD
    global inpt
    dataOk = 0
    clas = 'classe'
    pred = []

    # Read and parse the received data
    PacketBuffer = readData(Dataport)
    dataOk = parseData68xx(PacketBuffer)
    if dataOk > 0:
        # Calculate the probability of classes
        rt = model.predict(inpt, verbose=0)
        pred.append(float(rt[0][0]))
        print("Idle probability :",pred)
        if pred[0] > 1-THRESHOLD:
            clas = label[0]
        else:
            clas = label[1]
        print("Class :",clas,"\n")

    return dataOk, clas


# ------------------------------------------------------------------

# Infinite loop used in the plotting of the current detected class

def infinite_loop():
    start = time.process_time()
    dataOk = 0
    try:
        # Update the data and check if the data is okay
        dataOk, clas = update()

        if dataOk > 0:
            #print("Tempo de execucao: ")
            #print(time.process_time() - start)
            if clas == label[0]:
                imageLabel.config(image = greenB)
            elif clas == label[1]:
                imageLabel.config(image = orangeB)
            #elif clas == label[2]:
            #    imageLabel.config(image = redB)
            else:
                imageLabel.config(image = blackB)
            if DEBUG:
                print("Process time :",time.process_time() - start,"\n")
        imageLabel.after(1, infinite_loop)
    # Stop the program and close everything if Ctrl + c is pressed
    except KeyboardInterrupt:
        CLIport.write(('sensorStop\n').encode())
        CLIport.close()
        Dataport.close()
        window.destroy()
        return


# ------------------------------------------------------------------

# What to do if the exit button is pressed on the application

def exitProgram():
    CLIport.write(('sensorStop\n').encode())
    CLIport.close()
    Dataport.close()
    window.destroy()


# -------------------------    MAIN   -----------------------------------------  

 # Configurate the serial port
CLIport, Dataport = serialConfig(configFileName)

# Get the configuration parameters from the configuration file
configParameters = parseConfigFile(configFileName)

# Initialize the arrays
mat = np.zeros((NUMBER_OF_ROWS, NUMBER_OF_COLUMNS), dtype = np.float32)
res = np.zeros((NUMBER_OF_COLUMNS, NUMBER_OF_ROWS+1), dtype = np.uint16)
inpt = np.zeros((1, NUMBER_OF_ROWS, NUMBER_OF_COLUMNS, DEPTH, 1), dtype = np.float32)   # Input sample for the CNN

# Initialize the display window
window = tk.Tk()
window.title("First GUI")
window.geometry("240x300")

# Load the colors used to display class on the window
greenB = ImageTk.PhotoImage(Image.open(os.getcwd() + '\\Images\\greenB.png').resize((238,245)))
redB = ImageTk.PhotoImage(Image.open(os.getcwd() + '\\Images\\redB.png').resize((238,245)))
yellowB = ImageTk.PhotoImage(Image.open(os.getcwd() + '\\Images\\yellowB.png').resize((238,245)))
orangeB = ImageTk.PhotoImage(Image.open(os.getcwd() + '\\Images\\orangeB.png').resize((238,245)))
blackB = ImageTk.PhotoImage(Image.open(os.getcwd() + '\\Images\\blackB.png').resize((238,245)))

imageLabel = tk.Label(window)
imageLabel.pack()

# Exit button
exit_btn= tk.Button(window, text='Exit', width=12,height=2,bg='white',fg='black',command=exitProgram)
exit_btn.pack(side = 'bottom', pady = 3)


# Main loop 
def Main_Program():
    global model, model_name
    
    # Load CNN model
    model = models.load_model(model_name)

    infinite_loop()                         # Call for infinite display loop

Main_Program()                          # Call for main loop

window.mainloop()
