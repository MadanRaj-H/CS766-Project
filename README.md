## Table of Content
1. [Introduction](#introduction)
2. [Framework](#framework)
3. [Hand Segmentation](#hand-segmentation)
4. [CNN Model](#cnn-model)
5. [Results](#results)
6. [Conclusion and future work](#conclusion-and-future-work)
7. [References](#references)

### Introduction
<p>
Gesture recognition is an active research field that tries to integrate the gestural channel in Human-Computer interaction. It has a variety of applications like virtual environment control, sign language translation, robot remote control, musical creation[1], etc. Several applications have been built around human gestures, wherein the detected gesture triggers a command or serves as an input to the system. It has proven to be useful and enhance user experience in many scenarios.
</p>
<p>
A configurable framework for gesture recognition can let application developers easily incorporate gesture controls onto their system by mapping gestures to their corresponding actions depending upon their implementation logic. Applications can support sophisticated user interfaces with significantly less effort. Since the existing approaches require external equipment like gloves or fixed background to detect the hand, it makes the task of integrating with other applications difficult. So, a framework that detects gestures using only the raw input feed from the camera could be a promising tool for application development.
</p>

### Framework
<p>
Our goal of the project is to build a prototype for the described gesture recognition framework. Convolutional Neural Network is a deep learning technique whereby several convolutions and pooling layers are stacked to perform operations like transformation, feature extraction, and decision making. It is the state of the art algorithm for object recognition. The American Sign Language has 26 different gestures which are the symbols for each English alphabet. A dataset consisting of images depicting gestures for each of these 26 types could be used to build a CNN model that classifies a given gesture into one among these. This model serves as the backbone of our framework.
</p>
<p>
The raw input feed from the camera is split into frames. For every few frames where the change in the scene is minimal, one among them is taken as a reference and is pre-processed to identify the position of the hand in the scene, irrespective of the background. The region of interest is then passed on to the CNN model to predict the observed gesture.
</p>

### Hand Segmentation
<p>
Hand segmentation can be done using several approaches like motion-based, skin-based or object detection techniques. Each technique has its advantages in terms of complexity and accuracy. Most systems use glove based methods for better accuracy. Very few approaches have been proposed in terms of RGB for hand detection.
As you can see, there are a lot of challenges in detecting the hand from the input frame. Hence, we came up with an idea where we fix a boundary for the region of interest and update the background. 
</p>
<p>
The steps involved in detecting the hand in our framework is discussed below.
<li>OpenCV continuously gets input frames from the system camera.</li>
<li>From the input frame, we obtain the region of interest frame and convert it into a grayscale image and blur it using Gaussian.</li>
<li>The first ROI frame is saved as a reference for the background and in the next 25 frames, we accumulate the weights and update the background.</li>
<li>Once the background is fixed, perform morphological transformations like dilation, erosion and median filter to sieve the background noise and preserve structural integrity.</li>
<li>Threshold the image to get the foreground. The maximum contour area gives the segmented hand and sends it to the model for gesture recognition.</li>
</p>


### CNN Model
<p>
The model we are using currently is a Convolution neural net with the architecture as shown below:
</p>


### Results
<p>
</p>

### Conclusion and future work
<p>
</p>


### References
<p>
<ol>https://www.sciencedirect.com/topics/computer-science/gesture-recognition</ol>
<ol>https://www.kaggle.com/grassknoted/asl-alphabet</ol>
<ol>https://medium.com/@ksusorokina/image-classification-with-convolutional-neural-networks-496815db12a8</ol>
<ol>https://www.ripublication.com/ijaer18/ijaerv13n9_90.pdf</ol>
<ol>http://cs231n.stanford.edu/reports/2016/pdfs/214_Report.pdf</ol>
</p>




