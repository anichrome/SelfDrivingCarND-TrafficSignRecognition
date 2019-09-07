# Project: Traffic Sign Recognition
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


# Goals of the project

The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


In this work, I have used the approaches written by Pierre Sermanet and Yann LeCun in their paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks". 

[//]: # (Image References)

[image1]: ./images_for_report/input_dataset.png "Traning data"
[image2]: ./images_for_report/input_dat_histogram.png "Histogram of traning data"
[image3]: ./images_for_report/preprocessed_data.png "Training data after preprocessing"
[image4]: ./images_for_report/validation_accuracy.png "Validation Accuracy"
[image5]: ./images_for_report/german_traffic_signs.png "Test Images from German Traffic Signs"
[image6]: ./images_for_report/softmax_1.png "Softmax probabilities"
[image7]: ./images_for_report/softmax_2.png "Softmax Probabilities"



## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.


### Summary of the dataset used

The summary is calculated using python with the help of numpy.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

The figure below shows a small subset of the input dataset.

![alt text][image1]

The graph shows the distribution of traffic sign in the training set.

![alt text][image2]

### Design, train and test a model architecture

Before a model could be designed and the data be trained, I pre-processed the available data using techiniques mentioned in the paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" by Pierre Sermanet and Yann LeCun. 

In the first step, I converted the RGB image to Grayscale as the color does not contribute to learning of traffic signs as mentioned in  the paper. I used luma channel from YUV color space for this.

This channel is then processed first by local normalization and then by global normalization. Local normalization increases the contrast within the image, whereas global normalization centers all images around its mean.

To increase the variance of the dataset I added four additional images for each training image. The processing pipeline is like this:

* translate randomly by [-2, 2] pixel in any direction
* rotate by random degree [-15, 15] 
* scale in any direction by pixels in range [-2, 2]

The figure below shows a subset of the training data after preprocessing steps.

![alt text][image3]

#### Model Architecture

The model I used consisted of the following layers. I adapted the LeNet model with the recommendations from Pierre Sermanet and Yann LeCun :

| Layer         		|     Description	        					| 
|:--------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Layer 1: Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x108 	|
| Layer 1: RELU					|												|
| Layer 1: Max pooling	      	| 2x2 stride,  outputs 14x14x108 				|
| Layer 2: Convolution 3x3     	| 1x1 stride, valid padding, outputs 10x10x108 	|
| Layer 2: RELU					|												|
| Layer 2: Max pooling	      	| 2x2 stride,  outputs 5x5x108 				|
| Dropout Layer 1	      	| Keep Propability: 0.5 				|
| Dropout Layer 2	      	| Keep Propability: 0.5 				|
| Flatten and Combine	      	| outputs 57153600 x 1 				|
| Fully connected| | 
| Classifier Layer 1: Network	    | 57 Hidden Units       									|
| Classifier Layer 1: RELU		| 												|
| Classifier Layer 1: Dropout	      	| Keep Propability: 0.5 				|
| Classifier Layer 2: Network	    | 43 Hidden Units       									|									|
| Softmax cross entropy with logits				|         									|
| Loss operation						| reduce mean												|
| Optimizer						| AdamOptimizer learning rate 0.0002												|


#### Model Training

As a first step, I trained the preprocessed dataset with leNet architecture as mentioned in the excercises. It showed good results in the beginning, but I could quickly notice that it overfitted the data. Especially while testing on new images, it showed bias towards a particular image. 

I then folowed the approach mentioned by Pierre Sermanet and Yann LeCun in their paper to change the model architecture. This improved the performance of the network a lot. The details of my training is discussed in the table below.

| Epoch | Batch Size | Learn Rate |
|:--:|:--:|:--:|
|30|32|0,0002|

#### Model modifications

To achieve the required validation accuracy, I had to make several changes to the LeNet architecture. They are discussed below.

##### Local Normalization

In addition to the global normalization of training images, I introduced a local normalization which contributed to the performance of my network. The local normalization impoved the contrast of edges a lot, even if pictures histogram looked already widespread. This does not only help the human eye to recognize a picture but is also beneficial for the network as features are exposed.

##### Dropout

In my first approach with the LeNet model, I did not use the drouputs and I believe this was a major factor for overfitting of data. Due to this reason, I included two drop out layers after the convolutional layers and as it can be seen from the results, the data was not overfit anymore and the performance improved drastically as compared to the first approach.

#### Final Results
With the discussed parameters and the model, I could achieve the following results:

* validation set accuracy of 0.9927437641723356
* test set accuracy of 0.9736342042944116

The figure below shows the validation accuracy while training.

![alt text][image4]

### Test the Model on New Images

I downloaded the following 6 images from the internet to test the model I trained. The images I used can be seen here:

I chose the images which were relatively of lower resolution, having rotation and hard to identify on purpose. But, the model could identify them with a 100% accuracy on all of the images.

![alt text][image5]

### Softmax probabilities

The softmax probabilities of each image is as shown below.

![alt text][image6]
![alt text][image7]

As it can be seen, the model predicts the correct traffic sign with a higher softmax probability on all occasions.
