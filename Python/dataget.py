import serial
import time

Dataport = input("mmWave Demo output data port (standard port) = ")     #ask user for data port
Configport = input("mmWave Demo input config port (enhanced port) = ")  #ask user for config port
configFileName = input("Config file name = ")                           #ask user for Config file

#modified from kirkster96's github 
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
        print(i)
        time.sleep(0.01)
    return serconf, serdat

#Establish connection, sensor configuration and printing datas
serconf, serdat = serialConfig(configFileName)
while True :
    byteCount = serdat.inWaiting()         #get the count of bytes in buffer
    s = serdat.read(byteCount)             #read byteCount bytes from the buffer
    if s :
        print(s)