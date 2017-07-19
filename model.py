import csv
import os
import cv2
import numpy as np
import sklearn
import random

def BatchGenerator(image_folder, samples, batch_size):
    # Create generator to handle the large number of inputs without using too much memory
    nb_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, nb_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images, angles = [], []
            for batch_sample in batch_samples:
                img_center_name = image_folder + batch_sample[0].split('/')[-1]
                img_left_name = image_folder + batch_sample[1].split('/')[-1]
                img_right_name = image_folder + batch_sample[2].split('/')[-1]

                # Read all 3 images
                img_center = cv2.imread(img_center_name)
                img_left = cv2.imread(img_left_name)
                img_right = cv2.imread(img_right_name)

                # Get Steering angle for center image and extrapolate angle for left and right images
                angle_correction = 0.05
                steering_center = float(batch_sample[3])
                steering_left = steering_center + angle_correction
                steering_right = steering_center - angle_correction

                # Sdd images and angles to data set
                images.extend([img_center, img_left, img_right])
                angles.extend([steering_center, steering_left, steering_right])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def ReadCsvLogFile(csv_path):
    """Read the CSV driving log created from the Udacity Simulator."""
    samples = []
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        header = reader
        for idx, line in enumerate(reader):
            if idx > 0:
                samples.append(line)
    return samples

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
    for line in lines[1:]:
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

    return BuildNvidiaSelfDrivingModel(image_shape)

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
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Activation
    from keras.layers import Convolution2D, Cropping2D
    from keras.layers.pooling import MaxPooling2D
    from keras.layers.core import Lambda

    model = Sequential()
    model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=image_shape))
    model.add(Lambda(NormalizeImage))
    model.add(Convolution2D(24, 5, 5))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36,5,5))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48,5,5))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64,3,3))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    return model

def NormalizeImage(image):
    """Image Normalization"""
    return image / 255.0 - 0.5

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    print("--------------------------------------------")
    print("Project 3 - Behavioral Cloning")
    print("--------------------------------------------")

    csv_path = '../data/0/driving_log.csv'
    image_folder = '../data/0/IMG/'

    print("Loading Images from dataset")
    samples = ReadCsvLogFile(csv_path)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    print('Nb training samples: {}'.format(len(train_samples)))
    print('Nb validation samples: {}'.format(len(validation_samples)))

    img_shape = (160,320,3)
    print('Image Size: {}'.format(img_shape))

    # Create Generators
    train_generator = BatchGenerator(image_folder, train_samples, batch_size=32)
    validation_generator = BatchGenerator(image_folder, validation_samples, batch_size=32)

    # Build Model
    print("Building the model")
    model = BuildModel(img_shape)

    # Train
    print("Starting Training")
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

    # Save 
    print("Saving Model")
    model.save('model.h5')
