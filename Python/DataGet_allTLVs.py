import datetime
import os
import sys
import serial
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration file name
configFileName = os.getcwd() + '\\config_files\\config_file_doppler_azimuth.cfg'

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


# ------------------------------------------------------------------

filelocation = ''
tipo = 0

# Function to select data type to record and generate file location

def selectType():
    global filelocation, tipo
    os.system('clear')
    tipo = input("Please Input Class Name \n:>")
    #filelocation = os.getcwd() + '/DataSet/' +  tipo + '/' + tipo + '_'    #Raspberry's path to create a non existing file 
    filelocation = os.getcwd() + '\\DataSet\\' + tipo + '\\'                #Windows' path to create a non existing file
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

#Function to parse Datas from the read packet

def parseData68xx_AOP(byteBuffer):
    global res, compteur, mat, mat_ra_hm, QQ
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

    print("\nsample #",compteur,":")
    compteur += 1
    if DEBUG:
        print('numTLVs :',numTLVs)
    # Read the TLV messages
    for tlvIdx in range(numTLVs):
        # Check the header of the TLV message to find type and length of it
        tlv_type = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        tlv_length = np.matmul(byteBuffer[idX:idX + 4], word)
        idX += 4
        if DEBUG:
            print('\ntlv type :', tlv_type)
            print('tlv length :', tlv_length)

        # Read the data if TLV type 1 (Detected points) detected
        if tlv_type == 1:
            num_points = tlv_length/16
            if DEBUG:
                print(num_points,"points detected")
            vect = byteBuffer[idX:idX + tlv_length].view(np.uint32)     # Data vector
            points_array = np.zeros([int(num_points),4],dtype='uint32')
            points_array = vect.reshape(int(num_points),4)
            #if DEBUG:                                                  # Uncomment if there are no tlv type 7
                #labels = ['X[m]','Y[m]','Z[m]','Doppler[m/s]']
                #points_df = pd.DataFrame(points_array,columns=labels)
                #print(points_df)


        # Read the data if TLV type 5 (Doppler heatmap) detected
        if tlv_type == 5:
            resultSize = RANGE_FFT_SIZE * (DOPPLER_FFT_SIZE + 1) * np.dtype(np.uint16).itemsize
            if tlv_length == resultSize:
                if DEBUG:
                    print("Sizes Matches: ", resultSize)
                ares = byteBuffer[idX:idX + resultSize].view(np.uint16) # Data vector
                res = np.reshape(ares, res.shape)                       # Data array of the right size
                # Shift the data to the correct position
                rest = np.fft.fftshift(res, axes=(1,))      # put left to center, put center to right
                # Transpose the input data for better visualization
                result = np.transpose(rest)
                # Remove DC value from matrix
                mat = result[1:, :]
                if mat.shape == (DOPPLER_FFT_SIZE, RANGE_FFT_SIZE):
                    dataOK = 1
                    print('Range Doppler heatmap data :\n', pd.DataFrame(mat), '\n')
                else:
                    dataOK = 0
                    print("Invalid Range Doppler Matrix")
                    return dataOK, mat, QQ
                break
            else:
                dataOK = 0
                print("TLV length does not match expected size for Range Doppler data")
                return dataOK, mat, QQ
        
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
            print("\n",points_df,"\n")

        # Read the data if TLV type 8 (Range Azimuth Heatmap) detected
        if tlv_type == 8:
            expected_size = RANGE_FFT_SIZE * numTxAnt * numRxAnt * np.dtype(np.int16).itemsize * 2    # Expected TLV size : numRangebins * numVirtualAntennas * 2 bytes * 2 (Real + Imag values)
            print(expected_size)
            if tlv_length == expected_size:
                if DEBUG:
                    print("Sizes Matches: ", expected_size)
                vect_rahm = byteBuffer[idX:idX + tlv_length].view(np.int16)    # Data vector of the tlv value
                mat_ra_hm = np.reshape(vect_rahm,(RANGE_FFT_SIZE,numRxAnt*numTxAnt*2)) # Data array of the tlv value, real and imag values in 2 separated cells
                cmat_ra = np.zeros((RANGE_FFT_SIZE,numRxAnt*numTxAnt),complex)
                for n in range (RANGE_FFT_SIZE):                                # Reassembling real and imag values in one complex matrix
                    for m in range(0,numTxAnt*numRxAnt*2,2):
                        cmat_ra[n][m//2] = complex(mat_ra_hm[n][m+1],mat_ra_hm[n][m])
                #print('cmat:',cmat_ra.shape,pd.DataFrame(cmat_ra))
                Q = np.fft.fft(cmat_ra,n=DOPPLER_FFT_SIZE+1,axis=1)
                #print('Q:',Q.shape,pd.DataFrame(Q))
                QQ = np.fft.fftshift(abs(Q),axes=(1,))                          # Put left to center, put center to right
                #print('QQ:',QQ.shape,pd.DataFrame(QQ))
                QQ = QQ[:,2:]                                                   # Cut off first angle bin
                #print('QQ:',QQ.shape,pd.DataFrame(QQ))
                if QQ.shape == (RANGE_FFT_SIZE,DOPPLER_FFT_SIZE-1):
                    dataOK = 1
                    print('Range Azimuth Heatmap data :\n',pd.DataFrame(QQ),'\n')
                else:
                    dataOK = 0
                    print('Invalid Range Azimuth Matrix')
                    return dataOK, mat, QQ
            
            else:
                dataOK = 0
                print("TLV length does not match expected size for Range Azimuth data, check hard coded number of antennas")
                return dataOK, mat, QQ

        idX += tlv_length   # Check next TLV
    return dataOK, mat, QQ

# ------------------------------------------------------------------

# Funtion to update the data and display in the plot

def update():
    # Read and parse the received data
    PacketBuffer = readData(Dataport)           # Read a frame and store it in the packet Buffer
    PacketBufferLength = len(PacketBuffer)
    #ObjectsData = parser_one_mmw_demo_output_packet(PacketBuffer,PacketBufferLength,DEBUG)   # Parse statistics on objects detected, add later to gather more informatons
    dataOK, matu, mat_ra_hm = parseData68xx_AOP(PacketBuffer) # Parse Range-Doppler Heatmap
    return dataOK, matu, mat_ra_hm

# ------------------------------------------------------------------

# Funtion to save all the gathered heatmaps in separated csv files

def saveM(matf,n,type):
    global num
    count = 0
    if matf.shape[0] == n * num:
        for m in range(0, n * num, n):
            tm = datetime.datetime.now()
            name = f"{tm.year}_{tm.month}_{tm.day}_{tm.hour}_{tm.minute}_{tm.second}"
            f = open(filelocation + type + "\\" + name + "_" + tipo + "_" + str(count) + '.csv', 'w')
            np.savetxt(f, matf[m:m + n], fmt='%d', delimiter=' ')
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

selectType()

num = int(input("Number of samples: "))

# Configure the serial ports
CLIport, Dataport = serialConfig(configFileName)
 
# Get the configuration parameters from the configuration file
configParameters = parseConfigFile(configFileName)

# Initialize the arrays used in data parsing and saving
# Range-Doppler arrays
mat = np.zeros((DOPPLER_FFT_SIZE, RANGE_FFT_SIZE), dtype=np.float32)
res = np.zeros((RANGE_FFT_SIZE, DOPPLER_FFT_SIZE + 1), dtype=np.uint16)
matf = np.zeros((DOPPLER_FFT_SIZE * num, RANGE_FFT_SIZE), dtype=np.float32)
# Range-Azimuth arrays
QQ = np.zeros((RANGE_FFT_SIZE, DOPPLER_FFT_SIZE-1), dtype=np.int32)
mat_ra_hmf = np.zeros((RANGE_FFT_SIZE * num, DOPPLER_FFT_SIZE-1), dtype=np.float32)

def main():
    # Initializing loop vars
    count = 0 
    pos = 0
    pos_ra = 0
    # Main loop for the right amount of samples
    while True:
        try:
            # Update the data and check if the data is okay
            dataOk, matm, QQ = update()
            if DEBUG:
                print('Date of sample :', datetime.datetime.now())  # To check on sample frequency
            if dataOk > 0:
                if count != 0:
                    matf[pos:pos + DOPPLER_FFT_SIZE] = matm             # Matf will reassemble all Range-Doppler samples in one array for saveM call
                    pos += DOPPLER_FFT_SIZE
                    mat_ra_hmf[pos_ra:pos_ra + RANGE_FFT_SIZE] = QQ     # mat_ra_hmf will reassemble all Range-Azimuth samples in one array for saveM call
                    pos_ra += RANGE_FFT_SIZE
                count += 1
            if count == (num + 1):
                saveM(matf,DOPPLER_FFT_SIZE,"Doppler")                  # saving all samples in separated csv files
                saveM(mat_ra_hmf,RANGE_FFT_SIZE,"Azimuth")
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
