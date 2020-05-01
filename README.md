## Table of Content
1. [Introduction](#Introduction)
2. [Framework](#Framework)
3. [Hand Segmentation](#Hand-Segmentation)
4. [CNN Model](#CNN-Model)
5. [Results](#results)
6. [Conclusion and future work](#conclusion-and-future-work)
7. [References](#reference)

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

