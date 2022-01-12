from flask import Flask, Response, jsonify, render_template, request
import tensorflow as tf
import cv2
from videoframe import webcam
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)

