import serial
import time
import numpy as np

#import ti's frame parser function
from parser_mmw_demo import parser_one_mmw_demo_output_packet, parser_helper

#debug var to print operations
DEBUG = True

#initialize byte buffer and others useful vars
MAXBUFFERSIZE = 2**15
byteBuffer = np.zeros(MAXBUFFERSIZE,dtype = 'uint8')
byteBufferLength = 0
magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
byteword = []
wordread = 0
byteCount = 0
magicWordFound = False
nextMagicWordFound = False

Dataport = input("mmWave Demo output data port (standard port) = ")     #ask user for data port
Configport = input("mmWave Demo input config port (enhanced port) = ")  #ask user for config port
configFileName = input("Config file name = ")                           #ask user for Config file

#Function to configure the serial connection,  modified from kirkster96's github 
def serialConfig(configFileName): #Open the serial ports and configure the sensor, return Config Port and Dataport
    # Open the serial ports
    # Windows
    serdat = serial.Serial(Dataport, 921600)            #establishing data connection
    serconf = serial.Serial(Configport, 115200)         #establishing config connection
    if DEBUG :
        print("Data serial : ", serdat)
        print("Config serial : ", serconf)

    serconf.write(("configDataPort 921600 1").encode()) #Dataport configuration

    #Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        serconf.write((i+'\n').encode())
        time.sleep(0.01)
    return serconf, serdat

#Function to parse the data inside the configuration file, modified from kirkster96's github
def parseConfigFile(configFileName):
    configParameters = {} #Initialize an empty dictionary to store the configuration parameters
    
    #Read the configuration file
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        
        #Split the line word by word
        splitWords = i.split(" ")
        
        #Antenna config (Hard coded, find a way to detect antenna config)
        numRxAnt = 4        #4 Rx and 3 Tx for 15deg azimuth res and elevation, 4Rx and 2 Tx for 15deg azimuth res no elevation
        numTxAnt = 3
        
        #Parse utile infos in arguments of the profileCfg line
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1
            
            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2
                
            digOutSampleRate = int(splitWords[11])
            
        #Parse utile infos in arguments of the frameCfg line
        elif "frameCfg" in splitWords[0]:
            
            chirpStartIdx = int(splitWords[1])
            chirpEndIdx = int(splitWords[2])
            numLoops = int(splitWords[3])
            numFrames = int(splitWords[4])
            framePeriodicity = float(splitWords[5])

            
    #Combine the read data to obtain the configuration parameters           
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate)/(2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
    
    return configParameters
   


#Establish connection, sensor configuration and processing datas
serconf, serdat = serialConfig(configFileName)
configParameters = parseConfigFile(configFileName)
if DEBUG :
    print ("configParameters = ",configParameters)


#Loop to process frames using parser_one_mmw_demo_output_packet
for i in range (4) : #Number of processed frames, HARD-CODED !!!!!!!
    while not(nextMagicWordFound):                          #while two separated magic word (= 1 complete frame) have not been found
        wordread = serdat.read(serdat.inWaiting())          #get the read data in read buffer
        byteword = np.frombuffer(wordread, dtype = 'uint8')      #convert read data into uint8 which can be processed
        byteCount = len(byteword)                                #number of bytes in byte buf
        
        if (DEBUG and (len(byteword)!=0)) :
            print(byteword)

        # Check if limite size of buffer has been reached
        if (byteBufferLength + byteCount) < MAXBUFFERSIZE:
            byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteword[:byteCount]
            byteBufferLength = byteBufferLength + byteCount
        else :
            print('Error : Reached max buffer size')
            break
        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]   
        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc+8] # Gather the 8 following bytes of the possible location 
            if np.all(check == magicWord):# Check if it corresponds to magic word
                startIdx.append(loc)      # Append startIdx array with magic word locs
        if len(startIdx) == 1 :
            magicWordFound = True
        if len(startIdx) > 1 :
            nextMagicWordFound = True
    print(startIdx)
    #parse first frame found in buffer and recover utile datas
    (result, headerStartIndex, totalPacketNumBytes, numDetObj, numTlv, subFrameNumber, detectedX_array, detectedY_array, detectedZ_array, detectedV_array, detectedRange_array, detectedAzimuth_array, detectedElevAngle_array, detectedSNR_array, detectedNoise_array) = parser_one_mmw_demo_output_packet(byteBuffer,byteBufferLength,DEBUG)

    #delete already processed data from buffer
    tempByteBuffer = byteBuffer[headerStartIndex+totalPacketNumBytes:]  #recover remaining unprocessed datas in temp buffer
    byteBuffer = np.zeros(MAXBUFFERSIZE, dtype = 'uint8')               #reset main buffer
    byteBuffer = tempByteBuffer                                         #paste remaining unprocessed datas in main buffer
    byteBufferLength = len(byteBuffer)
    print('Frame processed, new buffer length = ', byteBufferLength)

    #reset loop condition (first magic word already found obviously, trouver un moyen de ne pas avoir a le check de nouveau)
    magicWordFound = False
    nextMagicWordFound = False
    startIdx = []
# !!!!!!!!!!!!!!!!!!!!!!!!!!! LOOP HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#i=0
#while i<5 :
#    byteCount = serdat.inWaiting()         #get the count of bytes in buffer
#    s = serdat.read(byteCount)             #read byteCount bytes from the buffer
#    if (s and DEBUG):                                 #printing datas gathered
#        print(s)
#        i+=1


serconf.write(("sensorStop").encode())  #stopping sensor before closing ports
serconf.close()                         #closing ports
serdat.close()
