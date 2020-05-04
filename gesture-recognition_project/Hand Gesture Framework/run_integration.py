from __future__ import absolute_import
from __future__ import division
import numpy as np
from flask import Flask, render_template, Response,jsonify
from imutils import face_utils
import cv2, imutils, dlib, numpy as np, os, tensorflow as tf, sys
import struct, collections
import projectlibrary as plib
from keras.preprocessing import image
from os import listdir
from os.path import isfile, join

mypath = "./asl_alphabet_test_processed/X"
files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
count = 0
for curfile in files:
  print(curfile)
  img = image.load_img(os.path.join(mypath,curfile),target_size=(224,224))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  predictions= plib.handclassify(img)
  letter,confidence=predictions[0]
  print("prediction"+letter)
  if letter is 'q':
    count = count+1

print(count)
