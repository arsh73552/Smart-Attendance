import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle

def generateEncodings(images):
    encodeList = []
    for img in images:
        print("hehe")
        token = face_recognition.face_encodings(img, num_jitters=10)[0]
        encodeList.append(token)
    return encodeList

Attendance = [False * 14]
images = []
classLabels = []
path = r'C:\Users\arsh0\OneDrive\Documents\Smart Attendance\IAmNotDoingThisAgain'
for img in os.listdir(path):
    trainImg = cv2.imread(os.path.join(path, img))
    images.append(trainImg)
    classLabels.append(img[0:2])
    print(img)

#encoded_face_train = generateEncodings(images)
#with open("test", "wb") as fp:
#    pickle.dump(encoded_face_train, fp)

with open("test", "rb") as fp:
    encoded_face_train = pickle.load(fp)

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.rotate(img, cv2.ROTATE_180)
    imgS = cv2.resize(img, (0,0), None, 0.1,0.1)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(imgS,number_of_times_to_upsample=3)
    print(faces_in_frame)
    print(type(faces_in_frame))
    encoded_faces_test = face_recognition.face_encodings(imgS, faces_in_frame)
    for i in range(len(encoded_faces_test)):
        faceDist = face_recognition.face_distance(encoded_face_train, encoded_faces_test[i])
        print(faceDist)
        matchIndex = np.argmin(faceDist)
        if(faceDist[matchIndex] < 0.5):
            name = classLabels[matchIndex]
            y1,x2,y2,x1 = faces_in_frame[i]
            # since we scaled down by 4 times
            y1, x2,y2,x1 = y1*10,x2*10,y2*10,x1*10
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0), cv2.FILLED)
            cv2.putText(img,name, (x1+6,y2-5), cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
    #cv2.imshow('webcam', img)
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

