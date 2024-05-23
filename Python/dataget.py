import serial
import time

Dataport = input("mmWave Demo output data port (standard port) = ")     #ask user for data port
Configport = input("mmWave Demo input config port (enhanced port) = ")  #ask user for config port
configFileName = input("Config file name = ")                           #ask user for Config file

#Function to configure the serial connection,  modified from kirkster96's github 
def serialConfig(configFileName): #Open the serial ports and configure the sensor, return Config Port and Dataport
    # Open the serial ports
    # Windows
    serdat = serial.Serial(Dataport, 921600)            #establishing data connection
    print("Data serial : ", serdat)
    serconf = serial.Serial(Configport, 115200)         #establishing config connection
    print("Config serial : ", serconf)

    serconf.write(("configDataPort 921600 1").encode()) #Dataport configuration

    # Read the configuration file and send it to the board
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
   

#Establish connection, sensor configuration and printing datas
serconf, serdat = serialConfig(configFileName)
configParameters = parseConfigFile(configFileName)
print ("configParameters = ",configParameters)
i=0
while i<5 :
    byteCount = serdat.inWaiting()         #get the count of bytes in buffer
    s = serdat.read(byteCount)             #read byteCount bytes from the buffer
    if s :
        print(s)
        i+=1
serconf.write(("sensorStop").encode())  #stopping sensor before closing ports
serconf.close()                         #closing ports
serdat.close()
