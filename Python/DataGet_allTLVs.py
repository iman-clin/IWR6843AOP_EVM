import datetime
import os
import sys
import serial
import time
import numpy as np

from parser_mmw_demo import parser_one_mmw_demo_output_packet

# Configuration file name
configFileName = os.getcwd() + '/config_file_gics.cfg'

# Number of rows and columns for TLV type 5
NUMBER_OF_ROWS = 63
NUMBER_OF_COLUMNS = 128

# Buffer and useful variables
CLIport = {}
Dataport = {}

maxBufferSize = 2**15
byteBuffer = np.zeros(maxBufferSize, dtype='uint8')
byteBufferLength = 0

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

filelocation = ''
tipo = 0

# Function to select data type to record and generate file location

def selectType():
    global filelocation, tipo
    os.system('clear')
    tipo = input("Please Input Class Name \n:>")
    #filelocation = os.getcwd() + '/DataSet/' +  tipo + '/' + tipo + '_'    #Raspberry's path to create a non existing file 
    filelocation = os.getcwd() + '\\DataSet\\' + tipo + '\\' + tipo + '_'   #Windows' path to create a non existing file
    if DEBUG:
        print('type location = ', filelocation)

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

#Function to extract Range-Doppler Heatmap Array from the read packet

def RangeDopplerHM(byteBuffer):
    global NUMBER_OF_COLUMNS, NUMBER_OF_ROWS
    global res, compteur, mat
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

        #if tlv_type == 1:                      # Add later, to gather information on detected points

        # Read the data if TLV type 5 detected
        if tlv_type == 5:
            if DEBUG:
                print("TYPE 5 TLV!!")
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
                # Remove DC value from matrix
                mat = result[1:, :]
                idX += tlv_length
                if mat.shape == (NUMBER_OF_ROWS, NUMBER_OF_COLUMNS):
                    dataOK = 1
                    print(mat, '\n')
                else:
                    dataOK = 0
                    print("Invalid Matrix")
                    return dataOK, mat
                break
            else:   # If the TLV is of other type than 5
                idX += tlv_length
    return dataOK, mat

# ------------------------------------------------------------------

# Funtion to update the data and display in the plot

def update():
    # Read and parse the received data
    PacketBuffer = readData(Dataport)           # Read a frame and store it in the packet Buffer
    PacketBufferLength = len(PacketBuffer)
    #ObjectsData = parser_one_mmw_demo_output_packet(PacketBuffer,PacketBufferLength,DEBUG)   # Parse statistics on objects detected, add later to gather more informatons
    dataOK, matu = RangeDopplerHM(PacketBuffer) # Parse Range-Doppler Heatmap
    return dataOK, matu, ObjectsData

# ------------------------------------------------------------------

# Funtion to save all the gathered heatmaps in separated csv files

def saveM():
    global matf, num
    count = 0
    if matf.shape[0] == NUMBER_OF_ROWS * num:
        for m in range(0, NUMBER_OF_ROWS * num, NUMBER_OF_ROWS):
            tm = datetime.datetime.now()
            name = f"{tm.year}_{tm.month}_{tm.day}_{tm.hour}_{tm.minute}_{tm.second}"
            f = open(filelocation + name + "_" + str(count) + '.csv', 'w')
            np.savetxt(f, matf[m:m + NUMBER_OF_ROWS], fmt='%d', delimiter=' ')
            count += 1
            f.close()
    else:
        print("Incorrect Size\n")
        print(matf.shape[0])
        CLIport.write(('sensorStop\n').encode())
        CLIport.close()
        Dataport.close()
        sys.exit()


# -------------------------    MAIN   ------------------------------

# Main loop      

selectType()

num = int(input("Number of samples: "))

# Configure the serial ports
CLIport, Dataport = serialConfig(configFileName)
 
# Get the configuration parameters from the configuration file
configParameters = parseConfigFile(configFileName)

mat = np.zeros((NUMBER_OF_ROWS, NUMBER_OF_COLUMNS), dtype=np.float32)
res = np.zeros((NUMBER_OF_COLUMNS, NUMBER_OF_ROWS + 1), dtype=np.uint16)
matf = np.zeros((NUMBER_OF_ROWS * num, NUMBER_OF_COLUMNS), dtype=np.float32)

def main():
    count = 0 
    pos = 0
    while True:
        try:
            # Update the data and check if the data is okay
            dataOk, matm, Objectsdata = update()
            if DEBUG:
                print('Date of sample :', datetime.datetime.now())
            if dataOk > 0:
                if count != 0:
                    matf[pos:pos + NUMBER_OF_ROWS] = matm
                    pos += NUMBER_OF_ROWS
                count += 1
            if count == (num + 1):
                saveM()
                CLIport.write(('sensorStop\n').encode())
                CLIport.close()
                Dataport.close()
                break
        
        # Stop the program and close everything if Ctrl + c is pressed
        except KeyboardInterrupt:
            CLIport.write(('sensorStop\n').encode())
            CLIport.close()
            Dataport.close()
            break

main()  # call for main sampling loop
