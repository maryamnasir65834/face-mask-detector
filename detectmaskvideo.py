# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	#Taking frame dimensions to create a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),(104.0, 177.0, 123.0))
	#Passing the blob through the network and obtaining face detections 
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	#Initialising face list, their corresponding locations, and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	#looping over detections
	for i in range(0, detections.shape[2]):
		#Extracting the probability associated with the detection
		confidence = detections[0, 0, i, 2]
		#Filtering out weak detections by ensuring the probability is greater than the minimum probability
		if confidence > 0.5:
			#Compute the (x, y)-coordinates of the bounding box for the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			#Extracting the face ROI, converting it from BGR to RGB channel ordering
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			#resizing to 224x224, and preprocessing
			face = img_to_array(face)
			face = preprocess_input(face)

			#Add face and bounding boxes to their respective lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	#Only make a prediction if at least one face is detected
	if len(faces) > 0:
		#Making batch predictions on all faces at the same time rather than one-by-one predictions
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	#Returning a 2-tuple of the face locations and their corresponding locations
	return (locs, preds)






