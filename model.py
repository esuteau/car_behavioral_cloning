import csv
import os
import cv2
import numpy as np

def GetDataSet(driving_log_csv_path, image_folder):
    """Read the CSV driving log created from the Udacity Simulator.
    You can define a specific folder indicating where the images are located.
    Returns a tuple (images, measurements)"""

    # TODO: Make it a generator, so that we don't load all the data at once.
    if not os.path.exists(driving_log_csv_path):
        raise Exception("Could not find the CSV log file.")

    if not os.path.exists(image_folder):
        raise Exception("Could not find the image data folder")

    # Read the CSV summary
    lines = []
    with open(driving_log_csv_path) as f:
        reader = csv.reader(f)
        for line in reader:
            lines.append(line)

    # Read the images
    images = []
    measurements = []
    for line in lines[1:100]:
        source_path = line[0]
        image_filename = source_path.split('/')[-1]
        image_path = image_folder + image_filename
        image = cv2.imread(image_path)
        images.append(image)

        measurement = {}
        measurement["steering"] = float(line[3])
        measurement["throttle"] = float(line[4])
        measurement["brake"] = float(line[5])
        measurement["speed"] = float(line[6])
        measurements.append(measurement)

    # Return Dataset
    X = np.array(images)
    y = np.array([meas["steering"] for meas in measurements])

    return [X, y]


def BuildModel(image_shape):
    """Builds the Neural Network Model using keras"""

    return BuildInitialTestModel(image_shape)

def BuildInitialTestModel(image_shape):
    """Initial Model used to test the platform"""
    from keras.models import Sequential
    from keras.layers import Flatten, Dense
    from keras.layers import Convolution2D
    model = Sequential()
    model.add(Flatten(input_shape=image_shape))
    model.add(Dense(1))

    return model


def BuildNvidiaSelfDrivingModel(image_shape):
    """Neural Net Model based on Nvidia's model in https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/"""
    # TODO: Implement Model
    return None


def PreProcessData(X, y):
    """Data preprocessing"""

    # Normalize Input Data
    x_normalized = np.array(X / 255.0 - 0.5)

    return [x_normalized, y]

if __name__ == "__main__":
    print("--------------------------------------------")
    print("Project 3 - Behavioral Cloning")
    print("--------------------------------------------")


    print("Loading Images from dataset")
    (X, y) = GetDataSet('../data/driving_log.csv', '../data/IMG/')
    img_shape = X.shape
    print("{} Images Loaded".format(len(X)))
    print("Input Shape: {}".format(img_shape))

    # Preprocess Data
    print("Pre-processing data")
    X, y = PreProcessData(X, y)

    # Build Model
    print("Building the model")
    model = BuildModel(img_shape[1:])

    # Train
    print("Starting Training")
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, validation_split=0.2, shuffle=True)

    # Save 
    print("Saving Model")
    model.save('model.h5')
