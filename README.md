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

---

### Jupyter Files

**preparing_dataset and preparing_dataset_1folder :**

The first code will create a `.npz` file from the local dataset and break it in training, validation and testing sets for the CNN training (4 frames for one sample). It will also print a sample to check if the preparation went well.

The second one will create a `.npz` from one class of the local dataset. It should be used to generate a test dataset that will be used for feature prediction.

**Doppler_heatmap_multiclass_training_local :**

Build a CNN model and train it based on the keras/tensorflow models. It will use the the dataset prepared as a `.npz` file to train, test and validate the model. Also plot the accuracy and loss of the model and test the prediction on a testing dataset.

[Google colab notebook for online training](https://drive.google.com/file/d/1OV4RE8pCYmDNoZVx6Gf-anxS9c_T9hl8/view?usp=sharing)

## Required Python packages

- **numpy:** for operations on arrays
- **pandas:** for more convenient array display
- **serial:** for serial connection establishment with the sensor
- **time:** for waits and file names
- **matplotlib:** for results plotting
- **keras/tensorflow:** for cnn construction and training
- **PIL and tkinter:** for real-time plotting of results