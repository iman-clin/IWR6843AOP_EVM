# IWR6843AOP_EVM

Git repo about IWR6843_AOP_EVM application for people detection by CNN
Contains a parsing code to acquire data from serial connection, a keras' CNN training model and a heatmap plotter for the acquired datas

## Code Description

### Python Files

**DataGet_allTLVs :**

This code will :\
    - Establish the serial connection with IWR684 on ports specified by user (cf. `mmwave_studio_user_guide` in Ressources folder to know how to configure serial connection and flash the sensor) \
    - Parse the config file and configure the sensor (config file path in `configFileName` var)\
    - Recover the data sent by the sensor on Data Port and store it in a buffer\
    - Parse the utile data (detected objects, Range/Doppler heatmap, Range/Azimut heatmap) and store it in local arrays\
    - Store the datas in separated `.csv` files, one file per frame, number of acquired frames is specified by user input, note that the dataset folders should have been created before (one folder/type, each type containing Azimuth and Doppler subfolders)

**Doppler_heatmap_plot :**

Recover the data contained in `.csv` files of a user-specified class, processes them and display the corresponding Range/Doppler heatmaps. Will also calculate the deployed energy on each frame.

**Azimuth_heatmap_plot :**

Recover the data contained in `.csv` files of a user-specified class, processes them and display the corresponding Range/Azimuth heatmaps.

***CNN realtime files :***

- **CNN_Doppler_realtime :**\
 Read and parse the data from the sensor to do real-time class prediction using a pre-trained model. This code will display an application window featuring a color display, each color corresponds to a class detection. This code should be used with a Doppler 3D CNN and config file Doppler only.

- **CNN_Doppler_Azimuth_realtime :**\
 Same purpose but meant to be used with a pre-trained 3D CNN for class prediction on moving features (Doppler: presence or non-presence detection). And a pre-trained 2D CNN for class prediction on static features (Azimuth: Object moved or room in idle configuration). This code should be used with a Doppler 3D CNN and an Azimuth 2D CNN and config file Doppler_Azimuth

- **CNN_Cat_realtime :**\
 Same purpose as Doppler Azimuth realtime but meant to be used with a pre-trained 2D CNN using as feature the concatenation of Doppler and Azimuth heatmaps. Not reliable for now even though the CNN training goes well.

- **CNN_Doppler_Azimuth_realtime_lowpower :**\
 Same purpose as Doppler Azimuth realtime but use a low power configuration of the sensor. If the detected class corresponds to no presence and a room untouched, the sensor is put in sleep mode for a random amount of time between 1 and 5 seconds. This should lower the power consumption and max the sensor lifespan in order to achieve class detection on a long period of time. This code needs a sensor flashed with the custom firmware avalaible in the binaries dir of this repository in order to work properly (modifies the idlePowerCycle and adds the resetDevice CLI commands)

- **CNN_Doppler_Azimuth_realtime_plotting :**\
 Add the display of both heatmaps in separated window as well as the class prediction, could be used for a user supervised class prediction.

---

### Jupyter Files

**preparing_dataset and preparing_dataset_1folder :**

The first code will create a `.npz` file from the local dataset and break it in training, validation and testing sets for the CNN training (4 frames for one sample). It will also print a sample to check if the preparation went well.

The second one will create a `.npz` from one class of the local dataset. It should be used to generate a test dataset that will be used for feature prediction.

Diverse versions of the preparing dataset code are available in order to make the trainings easier for all the different CNN types.

**multiclass training local codes :**

Build a CNN model and train it based on the keras/tensorflow models. It will use the the dataset prepared as a `.npz` file to train, test and validate the model. Also plot the accuracy and loss of the model and test the prediction on a testing dataset.

Diverse versions of the multiclass training are available in order to generate the correct pre-trained CNN for the different applications (Doppler only, Doppler/Azimuth, Concatenated features...).

**Standalone CNNs :**

Assembles all the previous files in a single one. Used to make calibration, dataset preparation, CNN training and real time class prediction more easy to use in a single workflow. This code should is still incomplete and features are yet to be added.

[Google colab notebook for online training](https://drive.google.com/file/d/1OV4RE8pCYmDNoZVx6Gf-anxS9c_T9hl8/view?usp=sharing)

## Required Python packages

- **numpy:** for operations on arrays
- **pandas:** for more convenient array display
- **serial:** for serial connection establishment with the sensor
- **time:** for waits and file names
- **matplotlib:** for results plotting
- **keras/tensorflow:** for cnn construction and training
- **PIL and tkinter:** for real-time plotting of results
