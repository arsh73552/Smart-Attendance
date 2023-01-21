from helper import *
import threading

present = 49 * [False]
images = []
classLabels = []
counter = 0
load_data(images, classLabels)
encoded_face_train = generate_encodings_train(images)

def make_change(img, faceEncoding, num, counter):
    faceDistanceFromTrain = face_recognition.face_distance(encoded_face_train, faceEncoding).tolist()
    minIndex = faceDistanceFromTrain.index(min(faceDistanceFromTrain))
    imgCopy = img
    print(faceDistanceFromTrain[minIndex])
    if(faceDistanceFromTrain[minIndex] <= 0.45):
        cv2.rectangle(imgCopy, (face[3], face[0]), (face[1], face[2]), (255, 0, 0), 2)
        cv2.putText(imgCopy, str(minIndex + 1), (face[3], face[0]- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        foundFace = imgCopy[(face[0] - 50):(face[2] + 50), (face[3] - 50):(face[1] + 50)]
        imgName = str(minIndex + 1) + "-" + str(faceDistanceFromTrain[minIndex]) + ".png"
        present[minIndex] = True
        cv2.imwrite(os.path.join(r'C:\Users\arsh0\OneDrive\Documents\Smart Attendance\Prayge', imgName), foundFace)
        counter += 1

start = time.time() #To calculate Runtime.
video_path = r'C:\Users\arsh0\OneDrive\Documents\Smart Attendance\VID_20221102_150009.mp4'
cap = cv2.VideoCapture(video_path)
threadList = []
success, img = cap.read()
fno = 0
while success:
    if fno % 10 == 0:
        face_pos = face_recognition.face_locations(img, model = "cnn")
        for face in face_pos:
            encoded_face_test = face_recognition.face_encodings(img, [face]) # Generate encodings for the faces found in the image.
            t1 = threading.Thread(target = make_change, args = (img, encoded_face_test[0], fno, counter, ))
            t1.start()
            threadList.append(t1)                
            for i in range(len(threadList)):
                threadList[i].join()
    fno += 1
    success, img = cap.read()

for i in range(len(threadList)):
    threadList[i].join()

for i in range(len(present)):
    if not present[i]:
        print(str(i + 1) + " is not present!")
end = time.time()
print(end - start) # RuntimeQ