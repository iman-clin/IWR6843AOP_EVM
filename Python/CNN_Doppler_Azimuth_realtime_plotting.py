import os
import sys
import serial
import time
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.interpolate as spi
from PIL import ImageTk, Image
from keras import models

# Configuration file name
configFileName = os.getcwd() + '\\config_files\\config_file_doppler_azimuth_32x256_3D.cfg'

# CNN 3d model, min and max values
model_name_dop = os.getcwd() + '\\all_targets_doppler_1037_3598.h5'
min_dop = 1045.0
max_dop = 3598.0

# CNN 2D model, min and max values
model_name_az = os.getcwd() + '\\all_targets_azimuth_0_10922.h5'
min_az = 0
max_az = 10922

THRESHOLD = 0.1                                # Threshold for non-idle probability

# Number of rows and columns for heatmap samples
DOPPLER_FFT_SIZE = 31
RANGE_FFT_SIZE = 256
DEPTH = 4

# Buffer and useful vars
CLIport = {}
Dataport = {}

maxBufferSize = 2**16
byteBuffer = np.zeros(maxBufferSize, dtype='uint8')
byteBufferLength = 0
init = 0

compteur = 0
magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

ObjectsData = 0

DEBUG = True

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
        numTxAnt = 2                # Number of azimuth antennas only

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
    if DEBUG:
        print(configParameters)

    return configParameters


# ------------------------------------------------------------------

# Function to normalize a matrix

def norm(mat,type):
    if type == 'Doppler':
        aux_n1 = np.subtract(mat, min_dop)
        aux_n2 = np.divide(aux_n1, (max_dop - min_dop))
        norm_mat = aux_n2.reshape(DOPPLER_FFT_SIZE,
                                RANGE_FFT_SIZE)
        return(norm_mat)
    if type == 'Azimuth':
        aux_n1 = np.subtract(mat, min_az)
        aux_n2 = np.divide(aux_n1, (max_az - min_az))
        norm_mat = aux_n2.reshape(RANGE_FFT_SIZE,
                                DOPPLER_FFT_SIZE)
        return (norm_mat)
    else:
        print('Invalid data type !')
        return(mat)

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
    global res, mat, inpt_dop, inpt_az
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

        # Read the data if TLV type 4 (Range Azimuth Heatmap) detected
        if tlv_type == 4:
            expected_size = RANGE_FFT_SIZE * numTxAnt * numRxAnt * np.dtype(np.int16).itemsize * 2    # Expected TLV size : numRangebins * numVirtualAntennas * 2 bytes * 2 (Real + Imag values)
            if tlv_length == expected_size:
                if DEBUG:
                    print("Sizes Matches: ", expected_size)
                vect_rahm = byteBuffer[idX:idX + tlv_length].view(np.int16)    # Data vector of the tlv value
                mat_ra_hm = np.reshape(vect_rahm,(RANGE_FFT_SIZE,numRxAnt*numTxAnt*2)) # Data array of the tlv value, real and imag values in 2 separated cells
                cmat_ra = np.zeros((RANGE_FFT_SIZE,numRxAnt*numTxAnt),complex)
                for n in range (RANGE_FFT_SIZE):                                # Reassembling real and imag values in one complex matrix
                    for m in range(0,numTxAnt*numRxAnt*2,2):
                        cmat_ra[n][m//2] = complex(mat_ra_hm[n][m+1],mat_ra_hm[n][m])
                Q = np.fft.fft(cmat_ra,n=DOPPLER_FFT_SIZE+1,axis=1)
                Q = abs(Q)                                                      # Magnitude of the fft
                Q = np.fft.fftshift(Q,axes=(1,))
                QQ = Q[:,1:]
                inpt_az = QQ.reshape(1,
                        RANGE_FFT_SIZE,
                        DOPPLER_FFT_SIZE,
                        1)
                if inpt_az.shape == (1,RANGE_FFT_SIZE,DOPPLER_FFT_SIZE,1):
                    dataOK = 1
                    if DEBUG == True:
                        print('Range Azimuth Heatmap data :',Q,'\n')
                else:
                    dataOK = 0
                    print('Invalid Range Azimuth Matrix')
                    return dataOK
            
            else:
                dataOK = 0
                print("TLV length does not match expected size for Range Azimuth data, check hard coded number of antennas")
                return dataOK

        # Read the data if TLV type 5 (Doppler heatmap) detected
        if tlv_type == 5:
            compteur += 1
            resultSize = RANGE_FFT_SIZE * (DOPPLER_FFT_SIZE + 1) * np.dtype(np.uint16).itemsize
            if tlv_length == resultSize:
                if DEBUG:
                    print("Sizes Matches: ", resultSize)
                    print("\nRange Bins: ", RANGE_FFT_SIZE)
                    print("\nDoppler Bins: ", DOPPLER_FFT_SIZE + 1)
                
                ares = byteBuffer[idX:idX + resultSize].view(np.uint16) # Data vector
                res = np.reshape(ares, res.shape)                       # Data array of the right size
                # Shift the data to the correct position
                rest = np.fft.fftshift(res, axes=(1,))      # put left to center, put center to right
                # Transpose the input data for better visualization
                result = np.transpose(rest)
                # Normalize the data
                mat = norm(result[1:],'Doppler')
                if mat.shape == (DOPPLER_FFT_SIZE, RANGE_FFT_SIZE):
                    if init < 4:
                        inpt_dop[:, : , : , init, 0] = mat
                        init += 1
                        dataOK = 0
                    else:
                        dataOK = 1
                        inpt_dop[:, : , : , 0, 0] = inpt_dop[:, : , : , 1, 0]
                        inpt_dop[:, : , : , 1, 0] = inpt_dop[:, : , : , 2, 0]
                        inpt_dop[:, : , : , 2, 0] = inpt_dop[:, : , : , 3, 0]
                        inpt_dop[:, : , : , 3, 0] = mat

                # Remove DC value from matrix
                mat = result[1:, :]
                if mat.shape == (DOPPLER_FFT_SIZE, RANGE_FFT_SIZE):
                    dataOK = 1
                    if DEBUG == True:
                        print('Range Doppler heatmap data:\n',mat,'\n')
                else:
                    dataOK = 0
                    print("TLV length does not match expected size for Range Doppler data, check Range and Doppler FFT sizes vars")
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

label = ['Idle', 'Presence', 'Object moved']    # Labels

# Funtion to update the data and display in the plot

def update():
    global model_dop, model_az, THRESHOLD
    global inpt_dop, inpt_az
    dataOk = 0
    clas = 'classe'
    pred_dop = []
    pred_az  = []

    # Read and parse the received data
    PacketBuffer = readData(Dataport)
    dataOk = parseData68xx(PacketBuffer)
    if dataOk > 0:
        # Calculate the probability of classes for 3D CNN
        rt_dop = model_dop.predict(inpt_dop, verbose=0)
        print("Class probabilities for moving features:",rt_dop)
        pred_dop.append(float(rt_dop[0][0]))
        # Calculate the probability of classes for 2D CNN
        rt_az = model_az.predict(inpt_az, verbose=0)
        print("Class probabilities for static features:",rt_az)
        pred_az.append(float(rt_az[0][0]))
        # Update class depending on idle probabilities
        print("Idle probabilities:\n",'Doppler:',pred_dop,'\n Azimuth:',pred_az)
        if pred_az[0] > 1-THRESHOLD and pred_dop[0] > 1-THRESHOLD:
            clas = label[0]
        elif pred_dop[0] < 1-THRESHOLD:
            clas = label[1]
        else:
            clas = label[2]
        print("Class :",clas,"\n")

        dop_hm = inpt_dop[0,:,:,3,0]
        az_hm = inpt_az[0,:,:,0]

        return dataOk, clas, dop_hm, az_hm
    
    else:
        # Return empty matrices if data not ok
        clas = 'Error'
        print('Acquisition failed')
        return dataOk, clas, np.zeros((RANGE_FFT_SIZE,DOPPLER_FFT_SIZE),dtype=np.float32), np.zeros((DOPPLER_FFT_SIZE,RANGE_FFT_SIZE),dtype=np.float32)



# ------------------------------------------------------------------

# Infinite loop used in the plotting of the current detected class

def infinite_loop():
    start = time.process_time()
    dataOk = 0
    try:
        # Update the data and check if the data is okay
        dataOk, clas, dop_hm, az_hm = update()

        if dataOk > 0:
            #print("Tempo de execucao: ")
            #print(time.process_time() - start)
            if clas == label[0]:
                imageLabel.config(image = greenB)
            elif clas == label[1]:
                imageLabel.config(image = orangeB)
            elif clas == label[2]:
                imageLabel.config(image = redB)
            else:
                imageLabel.config(image = blackB)
            if DEBUG:
                print("Process time :",time.process_time() - start,"\n")
            dop_plot(dop_hm)
            az_plot(az_hm)
        imageLabel.after(400, infinite_loop)
    # Stop the program and close everything if Ctrl + c is pressed
    except KeyboardInterrupt:
        CLIport.write(('sensorStop\n').encode())
        CLIport.close()
        Dataport.close()
        azimuth_win.destroy()
        doppler_win.destroy()
        window.destroy()
        return

# ------------------------------------------------------------------

# Plotting functions used to display current Doppler and Azimuth heatmaps

def dop_plot(dop_hm):
    global dop_canvas, dop_axes
    dop_axes.imshow(dop_hm, cmap='Spectral_r', interpolation='nearest', aspect='auto')
    dop_canvas.draw()
    return

def az_plot(az_hm):
    global az_canvas, az_axes
    zi = spi.griddata((posX, posY), az_hm.flatten(), (xiyi[0], xiyi[1]), method='linear')
    zi = zi.reshape(len(ylin), len(xlin))
    plt.contourf(xlin, ylin, zi, cmap='jet')
    az_canvas.draw()
    return

# ------------------------------------------------------------------

# What to do if the exit button is pressed on the application

def exitProgram():
    CLIport.write(('sensorStop\n').encode())
    CLIport.close()
    Dataport.close()
    azimuth_win.destroy()
    doppler_win.destroy()
    window.destroy()


# -------------------------    MAIN   -----------------------------------------  

 # Configurate the serial port
CLIport, Dataport = serialConfig(configFileName)

# Get the configuration parameters from the configuration file
configParameters = parseConfigFile(configFileName)

# Initialize the arrays
mat = np.zeros((DOPPLER_FFT_SIZE, RANGE_FFT_SIZE), dtype = np.float32)
res = np.zeros((RANGE_FFT_SIZE, DOPPLER_FFT_SIZE+1), dtype = np.uint16)
inpt_dop = np.zeros((1, DOPPLER_FFT_SIZE, RANGE_FFT_SIZE, DEPTH, 1), dtype = np.float32)   # Input sample for the CNN

inpt_az = np.zeros((1,RANGE_FFT_SIZE, DOPPLER_FFT_SIZE,1), dtype = np.float32)   # Input sample for the CNN

# Initialize the class display window
window = tk.Tk()
window.title("Class Display")
window.geometry("240x300")

# Initialize the Range Doppler heatmap display window and plot 
doppler_win = tk.Toplevel(window)
doppler_win.title("Doppler Range heatmap")
doppler_win.geometry("500x400")
dop_fig = plt.figure()                                          # Initialize the figure plot used to display the heatmap
dop_axes = dop_fig.add_subplot()
dop_canvas = FigureCanvasTkAgg(dop_fig, master=doppler_win)     # Extract a canvas from the plot that will be used to display the heatmap
dop_canvas.get_tk_widget().pack()
dop_canvas.draw()

# Initialize the Range Azimuth heatmap display window and plot 
    # Tk window initialization
azimuth_win = tk.Toplevel(window)
azimuth_win.title("Azimuth Range heatmap")
azimuth_win.geometry("500x400")
az_fig = plt.figure()
az_axes = az_fig.add_subplot()
az_canvas = FigureCanvasTkAgg(az_fig, master=azimuth_win)
az_canvas.get_tk_widget().pack()
az_canvas.draw()

    # Plot initialization
range_res = configParameters["rangeResolutionMeters"]
angle_bins, range_bins = DOPPLER_FFT_SIZE, RANGE_FFT_SIZE
range_depth = configParameters["numRangeBins"] * range_res
range_width, grid_res = range_depth/2, 400
theta = np.arcsin(np.linspace(-angle_bins / 2 + 1, angle_bins / 2 - 1, angle_bins) * (2 / angle_bins))
r = np.array(range(range_bins)) * range_res

posX = np.outer(r.T, np.sin(theta)).flatten()
posY = np.outer(r.T, np.cos(theta)).flatten()
xlin = np.linspace(-int(range_width), int(range_width), grid_res)
ylin = np.linspace(0, range_depth, grid_res)

xiyi = np.meshgrid(xlin, ylin)
az_axes.imshow(((0,)*grid_res,) * grid_res, cmap=plt.cm.jet, extent=[-range_width, +range_width, 0, range_depth], alpha=0.95)
az_axes.set_xlabel('Lateral distance along [m]')
az_axes.set_ylabel('Longitudinal distance along [m]')

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
    global model_dop, model_az, model_name_dop, model_name_az
    
    # Load CNN models
    model_dop = models.load_model(model_name_dop)
    model_az  = models.load_model(model_name_az)

    infinite_loop()                         # Call for infinite display loop

Main_Program()                          # Call for main loop


doppler_win.mainloop()
azimuth_win.mainloop()
window.mainloop()