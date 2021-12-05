

from flask import Flask, render_template, Response
import cv2
import os
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np


app=Flask(__name__)
cap = cv2.VideoCapture(0)
model = load_model("best_model.h5")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



def gen_frames():  
    while True:
        ret, test_img = cap.read()   # read the cap test_img
        if not ret:
            break
        else:
            gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

            faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

            for (x, y, w, h) in faces_detected:
                cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
                roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
                roi_gray = cv2.resize(roi_gray, (224, 224))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                predictions = model.predict(img_pixels)

                # find max indexed array
                max_index = np.argmax(predictions[0])

                emotions = ('Happy', 'xxxxx', 'SAD', 'Happy', 'sad', 'happy', 'xxxxx')
                predicted_emotion = emotions[max_index]

                cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            ret, buffer = cv2.imencode('.jpg', test_img)
            test_img = buffer.tobytes()
            yield (b'--test_img\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + test_img + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=test_img')
if __name__=='__main__':
    app.run(debug=True)