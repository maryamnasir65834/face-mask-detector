from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from detectmaskvideo import detect_and_predict_mask

def webcam():
	#Loading our face detector model
	prototxtPath = r"face_detector\deploy.prototxt"
	weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
	faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
	maskNet = load_model("maskdetector.model")

	#Initializing the video stream
	print("Starting video stream")
	vs = VideoStream(src=0).start()

	#Looping over the frames from the video stream
	while True:
		frame = vs.read()    
		frame = imutils.resize(frame, width=400)

		#Detect faces in the frame and determine if they are wearing a face mask or not
		(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
		for (box, pred) in zip(locs, preds):
			#Unpack bounding box and predictions
			(startX, startY, endX, endY) = box
			(mask, withoutMask) = pred

			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			#Including probability in the label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			#Displaying the label and bounding box rectangle on the output frame
			cv2.putText(frame, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

		#Showing the output frame
		ret, buffer =cv2.imencode('.jpg',frame)
		frame= buffer.tobytes() 
		yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
               