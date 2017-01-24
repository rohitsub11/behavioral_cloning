## Overview ##

This project is the behavioral cloning project to drive a car in a simulator.

## Files ##
1. model.py - The script used to create and train the model.
1. drive.py - The script to drive the car.
1. model.json - The model architecture.
1. model.h5 - The model weights.

## References ##

[Nvidia Paper] (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

## Method and model ##

I recorded my own images using Udacity simulator in training mode. However, Udacity had also provided their own training data which I used. 

The model that I started with is based on the Nvidia Paper. The model is as follows:

![alt tag](model_nvidia.png)
1. The first layer performs image normalization. **Update: removed this layer bacause it was overfitting my data.**
1. This is followed by 5 convolutional layers with relu activation for non linearity. The strided convolitions (first 3 convolutional layers) are 5x5 kernel with a 2x2 stride. The non-strided convolutions (last 2 convolutional layers) are 3x3 kernel with a 1x1 stride.
1. The convolutions are then followed by 3 fully connected layers leading to an output control value.

*UPDATE* I changed the model activations from relu to elu().


Dropout seems to be necessary to prevent the model from overfitting. I observed that a dropouts of .2 and .5 worked best through guess and check methodology and guidance from UDACITY as well as COMMA.AI.

## Image processing ##

The data had a lot of 0 steering angles. This would cause the model to train thinking that the road is straight and not know how to handle going off the road(say for example). To overcome this I added left and right images as well and added a small angle to the left camera and subtracted a small angle from the right camera. This is defined in the Udacity slides prior to the project as recovery.

There are 2 test tracks to train the model on. One is a brighter one whilst the other one is a relatively darker one. If we train the model only based on one track information, the model would not do well on the other track. To compensate for this we can either collect data from the other model or perform brightness augmentation to simulate day and night conditions.

After this and after reading many discussions on the forum as well as other papers, I cropped the image to remove some pixels from the top and bottom. to remove the horizon and hood of the car from the image. Then the question of resizing arose.

First try, I did not resize the image to see if they would go through the NVIDIA model without any problems. It seemed to at least run. However, I then resized the images to 66x200 so the match with the model requirements. I further resized the images to 64x64 to speed up the model training and yet not losing on accuracy or increasing loss.

