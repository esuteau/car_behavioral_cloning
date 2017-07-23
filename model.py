import csv
import os
import cv2
import numpy as np
import sklearn
import random

from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout
from keras.layers import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Lambda
from keras.preprocessing.image import img_to_array, load_img

import matplotlib.pyplot as plt

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

def GetImageShape():
    return (160, 320, 3)

def GetDataFromGenerator(image_folder, samples, batch_size=36):
    """Returns batches of center, left and right camera images using a generator.
        Randomly flips the images horizontally to remove steering bias and generalize the data."""
       
    nb_samples = len(samples)
    col_names = ['center', 'left', 'right']
    angle_correction = 0.15
    camera_correction = {'center': 0, 'left': angle_correction, 'right': -angle_correction}
    batch_size_triplet = batch_size // 3

    while 1:
        random.shuffle(samples)
        for offset in range(0, nb_samples, batch_size_triplet):
            batch_samples = samples[offset:offset+batch_size_triplet]

            images, angles = [], []
            for batch_sample in batch_samples:

                for col, camera in enumerate(col_names):

                    # Parse Windows/Linux Filenames
                    split_char = '/'
                    if '\\' in batch_sample[col]:
                        split_char = '\\'
                    img_name =  batch_sample[col].split(split_char)[-1]

                    # Load Image
                    image = load_img(image_folder + img_name)
                    image = img_to_array(image)

                    # Apply Steering Correction
                    steering_center = float(batch_sample[3])
                    steering = steering_center + camera_correction[camera]

                    # Apply Horizontal Flip Randomly on the data.
                    # Reverse the steering angle in this case
                    if random.random() > 0.5:
                        steering = -1*steering
                        image = cv2.flip(image, 1)

                    # Add image and angle to dataset
                    images.append(image)
                    angles.append(steering)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def PreProcessData(image, normalize=True, crop=True):
    """Pre-Process the data by Cropping and Normalizing. Expecting 160x320x3 images"""
    plot_img = False
    if plot_img:
        plt.imshow(image)
        plt.show()

    if crop:
        image = CropImage(image, 55, 25, 0, 0)

    if plot_img:
        plt.imshow(image)
        plt.show()

    if normalize:
        image = NormalizeImage(image)

    if plot_img:
        plt.imshow(image)
        plt.show()

    return image

def CropImage(image, crop_top_px, crop_bottom_px, crop_left_px, crop_right_px):
    """Crop an image. Expecting a 3 dimensional numpy array."""
    height = image.shape[0]
    width = image.shape[1]
    image = image[crop_top_px:width-crop_bottom_px, crop_left_px:width-crop_right_px, :]
    return image

def NormalizeImage(image):
    """Image Normalization"""
    return image / 255.0 - 0.5

def BuildModel(image_shape):
    """Builds the Neural Network Model using keras"""

    return BuildNvidiaSelfDrivingModel(image_shape)

def BuildInitialTestModel(image_shape):
    """Initial Model used to test the platform"""
    model = Sequential()
    model.add(Flatten(input_shape=image_shape))
    model.add(Dense(1))

    return model


def BuildNvidiaSelfDrivingModel(image_shape):
    """Neural Net Model based on Nvidia's model in https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/"""
    model = Sequential()
    model.add(Lambda(NormalizeImage, input_shape=image_shape))
    model.add(Cropping2D(cropping=((55,25), (0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu", input_shape=image_shape))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def BuildFinalModel(image_shape):
    """Neural Net Model based on Nvidia's model in https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/"""
    model = Sequential()
    model.add(Lambda(NormalizeImage, input_shape=image_shape))
    model.add(Cropping2D(cropping=((55,25), (0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Dense(128))
    model.add(Dense(64))
    model.add(Dense(1))
    return model


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    print("--------------------------------------------")
    print("Project 3 - Behavioral Cloning")
    print("--------------------------------------------")

    csv_path = '../data/driving_log.csv'
    image_folder = '../data/IMG/'

    print("Loading Images from dataset")
    samples = ReadCsvLogFile(csv_path)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    img_shape = GetImageShape()
    print('Nb training samples: {}'.format(len(train_samples)))
    print('Nb validation samples: {}'.format(len(validation_samples)))
    print('Image Size: {}'.format(img_shape))

    # Hyper Parameters:
    batch_size = 36

    # Create Generators
    train_generator = GetDataFromGenerator(image_folder, train_samples, batch_size=batch_size)
    validation_generator = GetDataFromGenerator(image_folder, validation_samples, batch_size=batch_size)

    # Build Model
    print("Building the model")
    model = BuildModel(img_shape)

    # Train
    print("Starting Training")
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
     validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

    # Save 
    print("Saving Model")
    model.save('model.h5')
