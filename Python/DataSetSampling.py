import datetime
import os
import sys
import serial
import time
import numpy as np
from numpy import loadtxt
#import tkinter as tk
from PIL import ImageTk, Image
from scipy import signal
import math

#Change the configuration file name to suit yours
configFileName = os.getcwd() + '/config_file_64x64.cfg'

#Number of rows and columns of a frame
NUMBER_OF_ROWS = 63
NUMBER_OF_COLUMNS = 64

#Buffer and utile variables
CLIport = {}
Dataport = {}
byteBuffer = np.zeros(2**15,dtype = 'uint8')
byteBufferLength = 0
DEBUG = True
contador = 0

# ------------------------------------------------------------------

# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName):
    
    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports
    
    try:
        CLIPortCheck = 0
        DataPortCheck = 0
        
        
        # Raspberry pi   /    Windows 
        while CLIPortCheck < 10:
            #CLIport = serial.Serial('/dev/ttyUSB0', 115200)                            #establish config connection on Raspberry pi
            # next 2 lines for connection on Windows
            configportname = input("mmWave Demo input config port (enhanced port) = ")  #ask user for config port
            CLIport = serial.Serial(configportname, 115200)                             #establish config connection on Windows
            if CLIport.is_open:
                break
            else :
                CLIPortCheck += 1
        if CLIPortCheck > 9:
            print("Couldn't connect to CLI Port!")
            sys.exit()

        while DataPortCheck < 10:
            #Dataport = serial.Serial('/dev/ttyUSB1', 921600)
            # next 2 lines for connection on Windows
            dataportname = input("mmWave Demo input config port (enhanced port) = ")    #ask user for data port
            Dataport = serial.Serial(dataportname, 921600)                              #establish data connection on Windows
            if Dataport.is_open:
                break
            else :
                DataPortCheck += 1
        if DataPortCheck > 9:
            print("Couldn't connect to Data Port!")
            sys.exit()
        
        # Read the configuration file and send it to the board
        config = [line.rstrip('\r\n') for line in open(configFileName)]
        for i in config:
            CLIport.write((i+'\n').encode())
            if DEBUG:
                print(i)
            time.sleep(0.01)
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
    
    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        
        # Split the line
        splitWords = i.split(" ")
        
        # Hard code the number of antennas, change if other configuration is used
        global numRxAnt
        numRxAnt = 4
        global numTxAnt
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
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2
                
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
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate)/(2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
    NUMBER_OF_COLUMNS = int(configParameters["numRangeBins"])
    NUMBER_OF_ROWS = int(configParameters["numDopplerBins"] - 1)
    
    return configParameters
   
# ------------------------------------------------------------------

# Compute power of two greater than or equal to `n`
def findNextPowerOf2(n):
 
    # decrement `n` (to handle cases when `n` itself
    # is a power of 2)
    n = n - 1
 
    # do till only one bit is left
    while n & n - 1:
        n = n & n - 1       # unset rightmost bit
 
    # `n` is now a power of two (less than `n`)
 
    # return next power of 2
    return n << 1
    

# Funtion to read and parse the incoming data
def readAndParseData68xx(Dataport, configParameters):
    global NUMBER_OF_COLUMNS, NUMBER_OF_ROWS
    global byteBuffer, byteBufferLength
    global mat, res, contador
    
    #start = time.process_time()
    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12;
    BYTE_VEC_ACC_MAX_SIZE = 2**15;
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1;
    MMWDEMO_UART_MSG_RANGE_PROFILE   = 2;
    MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5;
    maxBufferSize = 2**15;
    tlvHeaderLengthInBytes = 8;
    pointLengthInBytes = 16;
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
    
    # Initialize variables
    magicOK = 0 # Checks if magic number has been read
    dataOK = 0 # Checks if the data has been read correctly
        
    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype = 'uint8')
    byteCount = len(byteVec)
    
    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount
        
    # Check that the buffer has some data
    if byteBufferLength > 16384:
        
        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc+8]
            if np.all(check == magicWord):
                startIdx.append(loc)
               
        # Check that startIdx is not empty
        if startIdx:
            
            # Remove the data before the first start index
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength-startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength-startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength-startIdx[0]:]),dtype = 'uint8')
                byteBufferLength = byteBufferLength - startIdx[0]
                
            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0
                
            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2**8, 2**16, 2**24]
            
            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12+4],word)
            
            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1
    
    # If magicOK is equal to 1 then process the message
    if magicOK:
        # word array to convert 4 bytes to a 32 bit number
        word = [1, 2**8, 2**16, 2**24]
        
        # Initialize the pointer index
        idX = 0
        
        # Read the header
        magicNumber = byteBuffer[idX:idX+8]
        idX += 8
        version = format(np.matmul(byteBuffer[idX:idX+4],word),'x')
        idX += 4
        totalPacketLen = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        platform = format(np.matmul(byteBuffer[idX:idX+4],word),'x')
        idX += 4
        frameNumber = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        timeCpuCycles = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        numDetectedObj = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        numTLVs = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4
        subFrameNumber = np.matmul(byteBuffer[idX:idX+4],word)
        idX += 4


        # Read the TLV messages
        for tlvIdx in range(numTLVs):
            

            # Check the header of the TLV message
            tlv_type = np.matmul(byteBuffer[idX:idX+4],word)
            idX += 4
            tlv_length = np.matmul(byteBuffer[idX:idX+4],word)
            idX += 4

            # Read the data depending on the TLV message
            if tlv_type == MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP:
            
                if DEBUG:
                    print("TYPE 5 TLV!!")
                    print(contador)
                    contador += 1

                resultSize = NUMBER_OF_COLUMNS * (NUMBER_OF_ROWS + 1) * np.dtype(np.uint16).itemsize
                
                if(tlv_length == resultSize):
                
                    if DEBUG:
                        print("Sizes Matches: ")
                        print(resultSize)
                        print("\n")
                        print("Range Bins: ")
                        print(NUMBER_OF_COLUMNS)
                        print("\n")
                        print("Doppler Bins: ")
                        print(NUMBER_OF_ROWS + 1)
                        print("\n")
                
                    ares = byteBuffer[idX:idX+resultSize].view(np.uint16)
                    res = np.reshape(ares, res.shape)
                    
            
                # Shift the data to the corret position
                rest = np.fft.fftshift(res, axes=(1,))  # put left to center, put center to right
                # Transpose the input data for better visualization
                result = np.transpose(rest)
                # Remove DC value from matrix
                mat = result[1:,:]
                if mat.shape == (NUMBER_OF_ROWS, NUMBER_OF_COLUMNS):
                    dataOK = 1
                else:
                    dataOK = 0
                    print("Matriz Invalida!!")
                    return dataOK, mat
       
        
        #Remove already processed data
        if idX > 0 and byteBufferLength>idX:
            shiftSize = totalPacketLen
            
            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),dtype = 'uint8')
            byteBufferLength = byteBufferLength - shiftSize
            
            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0       

    return dataOK, mat
# ------------------------------------------------------------------

# Funtion to update the data and display in the plot
def update():
    dataOk = 0
      
    # Read and parse the received data
    dataOk, matu = readAndParseData68xx(Dataport, configParameters)
        
    return dataOk, matu


# -------------------------------------------------------------------------  
filelocation = ''
tipo = 0
def selectType():
    global filelocation, tipo
    os.system('clear')
    tipo = input("Please Input Class Name \n:>")
    #filelocation = os.getcwd() + '/DataSet/' +  tipo + '/' + tipo + '_'     #Raspberry's path to create a non existing file 
    filelocation = os.getcwd() + '\\DataSet\\' +  tipo + '\\' + tipo + '_'     #Windows' path to create a non existing file 
    print(filelocation)

        
def saveM():
    global matf, num
    count = 0
    if matf.shape[0] == NUMBER_OF_ROWS*num:
        for m in range(0, NUMBER_OF_ROWS*num, NUMBER_OF_ROWS):
            tm = datetime.datetime.now()
            name = str(tm.year) + "_" + str(tm.month) + "_" + str(tm.day) + "_" + str(tm.hour) + "_" + str(tm.minute) + "_" + str(tm.second)
            f = open(filelocation + name + "_" + str(count) + '.csv', 'w')
            np.savetxt(f, matf[m:m+NUMBER_OF_ROWS], fmt='%d', delimiter= ' ')
            count += 1
            f.close()
    else:
        print("Incorrect Size\n")
        print(matf.shape[0])
        CLIport.write(('sensorStop\n').encode())
        CLIport.close()
        Dataport.close()
        sys.exit()


# -------------------------    MAIN   -----------------------------------------  
# Main loop      
selectType()

num = int(input("Number of samples: "))
    
# Configurate the serial port
CLIport, Dataport = serialConfig(configFileName)

# Get the configuration parameters from the configuration file
configParameters = parseConfigFile(configFileName)

mat = np.zeros((NUMBER_OF_ROWS, NUMBER_OF_COLUMNS), dtype = np.float32)
res = np.zeros((NUMBER_OF_COLUMNS, NUMBER_OF_ROWS+1), dtype = np.uint16)
matf = np.zeros((NUMBER_OF_ROWS*num, NUMBER_OF_COLUMNS), dtype = np.float32)

def main():
    count = 0
    pos = 0
    while True:
        dataOk = 0
        try:
            # Update the data and check if the data is okay
            dataOk, matm = update()
            if dataOk > 0:
                if count != 0:
                    matf[pos:pos+NUMBER_OF_ROWS] = matm
                    pos += NUMBER_OF_ROWS
                count += 1
            if count == (num + 1):
                    saveM()
                    CLIport.write(('sensorStop\n').encode())
                    CLIport.close()
                    Dataport.close()
                    break
                        
        #time.sleep(0.05) # Sampling frequency of 30 Hz
            
        # Stop the program and close everything if Ctrl + c is pressed
        except KeyboardInterrupt:
            CLIport.write(('sensorStop\n').encode())
            CLIport.close()
            Dataport.close()
            break
        
main()
