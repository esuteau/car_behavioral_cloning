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

def PlotDataForDebugging(image_folder, samples):
    """Plot data for the debuggin"""
       
    nb_samples = len(samples)
    col_names = ['center', 'left', 'right']
    random.shuffle(samples)
    images = []

    for n in range(0, 2):
        batch_sample = samples[n]
        for col, camera in enumerate(col_names):

            # Parse Windows/Linux Filenames
            split_char = '/'
            if '\\' in batch_sample[col]:
                split_char = '\\'
            img_name =  batch_sample[col].split(split_char)[-1]

            # Load Image
            image = cv2.imread(image_folder + img_name)
            images.append(image)

    # PLot Original Images
    plt.figure()
    for n_image, image in enumerate(images):

        plt.subplot(2, 3, n_image+1)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


    # Flip if necessary Original Images
    plt.figure()
    images_flip = []
    for n_image, image in enumerate(images):

        # Apply Horizontal Flip Randomly on the data.
        # Reverse the steering angle in this case
        flip_img = random.random() > 0.5
        if flip_img:
            image = cv2.flip(image, 1)
        images_flip.append(image)

        plt.subplot(2, 3, n_image+1)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Crop Images
    plt.figure()
    for n_image, image in enumerate(images_flip):
        # Simulate cropping, just for the writeup
        image_cropped = image[65:135,:,:]

        plt.subplot(2, 3, n_image+1)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB))

    # Show all figures
    plt.show()

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

def NormalizeImage(image):
    """Image Normalization"""
    return image / 255.0 - 0.5

def BuildModel(image_shape):
    """Neural Net Model based on Nvidia's model in https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/"""
    model = Sequential()
    model.add(Lambda(NormalizeImage, input_shape=image_shape))
    model.add(Cropping2D(cropping=((65,25), (0,0))))
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

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    print("--------------------------------------------")
    print("Project 3 - Behavioral Cloning")
    print("--------------------------------------------")

    csv_path = '../data/combined/driving_log.csv'
    image_folder = '../data/combined/IMG/'

    print("Loading Images from dataset")
    samples = ReadCsvLogFile(csv_path)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    img_shape = GetImageShape()
    print('Nb training samples: {}'.format(len(train_samples)))
    print('Nb validation samples: {}'.format(len(validation_samples)))
    print('Image Size: {}'.format(img_shape))

    # For Debug only
    #train_data = PlotDataForDebugging(image_folder, train_samples)

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
