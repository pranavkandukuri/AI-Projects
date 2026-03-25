# AI-Projects
1. Face Mask Detection with Live Alert System
Abstract

Detects whether a person is wearing a mask using a webcam.

Introduction

Used during COVID-19 to ensure safety in public places.

Tools Used

Python, OpenCV, TensorFlow/Keras

Steps
Load dataset
Train CNN model
Detect faces via webcam
Predict mask/no mask
Code (Basic Version)
import cv2
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("mask_model.h5")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (100,100))/255.0
        face = np.reshape(face, (1,100,100,3))

        pred = model.predict(face)
        label = "Mask" if pred[0][0] > 0.5 else "No Mask"

        cv2.putText(frame, label, (x,y-10), 0, 1, (0,255,0), 2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("Mask Detection", frame)
    if cv2.waitKey(1) == 27:
        break
Conclusion

Works effectively in real-time detection.


PROJECT REPORT 2
AI Virtual Career Counsellor

Abstract:
A chatbot that suggests career options based on user interests.

Introduction:
Choosing a career is difficult. This chatbot helps students by recommending suitable careers.

Tools Used:
Python, NLTK, Rasa, Streamlit

Steps Involved:

Created intents
Preprocessed text
Trained chatbot
Built UI
Tested responses

Code:
def career_bot(user_input):
    if "tech" in user_input:
        return "You can go for Software Engineering or Data Science."
    elif "commerce" in user_input:
        return "You can choose CA, MBA or Finance."
    else:
        return "Please specify your interest."

while True:
    user = input("You: ")
    print("Bot:", career_bot(user))

Conclusion:
The system provides useful career suggestions and improves decision-making.

