#**Behavioral Cloning, by Erwan Suteau** 

##Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./CNN_Model_Architecture.png "Model Visualization"
[image2]: ./original_images.png "From Left to Right: Center, Left and Right Camera images"
[image3]: ./flipped_images.png "Images after random mirroring"
[image4]: ./cropped_images.png "Images after cropping"

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The overall model architecture is based on Nvidia's model, described here: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
My initial idea was to start with this model and add more convolutional layers, or increase the size of the fully connected layers if necessary.

But in the end, with enough training data and a couple of pre-processing techniques, this model runs really well on track 1.
It also trains really quickly on an AWS instance.

I played around with other CNN architectures but did not find a truly valid reason to move away from this model.

The NVIDIA model consists of 5 Convolutionnal feature maps and then 3 Fully connected layers.
Since we want to predict the steering angle, I added another fully-connected layer of size 1.

Here is a description of my final architecture architecture:

![alt text][image1]


####2. Attempts to reduce overfitting in the model

I did not use any dropout layer to reduce overfitting, but I generalized the training data by randomly flipping horizontally the data recorded on track 1.
I used the sample training + 4-5 laps at slow speed that using a PS4 controller to help record a really good training set.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

The training data used was a combination of the left, center and right camera images.
I used a correction factor for both left and right images to teach the network how to get the car back at the center of the road.

###Model Architecture and Training Strategy

####1. Solution Design Approach

As explained above, I started out with Nvidia's CNN model, without any pre-processing of the data and only the center images.
This wasn't a real success :(

After adding Normalizing the input data and cropping, the model made significant improvements, but the car still went for a swim right before the bridge.

The breakthrough happened after adding the left and right images with a steering correction factor and randomly flipping the input data.
I decided to do random image flipping to help the model generalize better. Track contains a lot left turns than right ones, and I did not want to record laps in the other direction.

After doing this the car was able to get across the bridge, but was still failing with the sharp turn that does not contain a side line.
At that point I was still training with the image samples, so I decided to almost triple this input dataset.

I recorded 4-5 new laps at low speed, using a PS4 controller to get some really clean data.
Adding this data to the initial set allowed the car to make its full first lap autonomously.

I tuned the steering angle to avoid looking like I was simulating drunk driving around the track, and I let the car run for a few laps to ensure it would not go swimming anymore.


####2. Final Model Architecture

The final model architecture is described in section 1.

####3. Creation of the Training Set & Training Process

As explained above, my process for the creation of the dataset was the following:
1. Record 4-5 additional laps at slow speed with a controller to ensure a clean training set.
2. Use Left-Center-Right cameras with correction steering factors to help the car get back to the center of the road.
3. Flip the images randomly to help the model generalize.
4. Shuffle the images and split the dataset: 80% training, 20% validation
5. Normalize the images and crop the top 65 pixels and bottom 25 pixels to remove the sky and the hood of the car.

My complete dataset contains close to 54,000 images (combining left, right, center), with 80% going to training and 20% to validation.
I processed the data set in batches of 36 images.
A total of 3 epochs and an Adam optimizer worked well for training.

This below is an example of the 2 sets of center-left-right camera images, before and after processing.

Original Images (From left to right: Center, Left and Right Cameras)

![alt text][image2]

Images After Random Mirroring (Top-Left and Top-Right are mirrored)

![alt text][image3]

Images After Cropping

![alt text][image4]