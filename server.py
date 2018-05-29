import numpy as np
from matplotlib import pyplot as plt
import os
from keras.layers import Dense, Convolution2D, UpSampling2D, MaxPooling2D, ZeroPadding2D, Flatten, Dropout, Reshape
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Conv2D
import cv2
from werkzeug import secure_filename 

from flask import Flask, Response, request
app = Flask(__name__)


model = Sequential()

model.add(Conv2D (64, (3, 3) , input_shape=(48, 48 , 1) ,activation='relu'))
model.add(Conv2D (64, (3, 3) , activation='relu'))
model.add(Conv2D (64, (3, 3) , activation='relu'))
model.add(Conv2D (64, (3, 3) , activation='relu'))
model.add(Conv2D (64, (3, 3) , activation='relu'))


model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.load_weights("face_model.h5")


@app.route('/predict', methods = ['POST'])	
def predict():
	f = request.files['image'] # key
	f.save(secure_filename('test.jpeg'))
	new = cv2.imread('test.jpeg' , 0)
	scale1 = 1.86
	scale2 = 2.47
	width = int(new.shape[1]*float(scale2)/100)
	height = int(new.shape[0]*float(scale1)/100)
	dim = (width , height)

	resized = cv2.resize(new , dim , interpolation = cv2.INTER_AREA)
	resized = resized.reshape((1 ,48, 48 , 1))
	prediction = model.predict(resized)
	ans = np.argmax(prediction)
	dic = { 0 : "ANGRY" , 1 : "DISGUST" , 2 : "FEAR" , 3 : "HAPPY" , 4 : "SAD" , 5 : "SUPRISE" , 6 : "NEUTRAL" }
	final = {"HAPPY" : 1 , "SUPRISE" : 2 , "NEUTRAL" : 3 , "FEAR" : 4 , "ANGRY" : 5 , "DISGUST" : 6 , "SAD" : 7}
	result = str(final[dic[ans]])
	tmpo= str(final[dic[ans]])+ "\nAngry = "+ str(prediction[0][0])+ "\nDisgust = "+ str(prediction[0][1])+ "\nFear = "+str(prediction[0][2]) + "\nHappy = " + str(prediction[0][3]) + "\nSad = " + str(prediction[0][4]) + "\nSuprise = " + str(prediction[0][5]) + "\nNeutral = " + str(prediction[0][6]) 
	return Response(response=tmpo, status=200)