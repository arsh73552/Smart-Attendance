from helper import *

images = []
classLabels = []
load_data(images, classLabels)
savePath = r'C:\Users\arsh0\OneDrive\Documents\Smart Attendance\Prayge' #Path to IMG with Label and a random garbage value
currPath = r'C:\Users\arsh0\OneDrive\Documents\Smart Attendance' #Working directory.
encoded_face_train = generate_encodings_train(images)
start = time.time() #To calculate Runtime.
path = r'C:\Users\arsh0\OneDrive\Documents\Smart Attendance\Face Detected' #Faces detected using YOLO SSD.
video_path = r'C:\Users\arsh0\OneDrive\Documents\Smart Attendance\VID_20221102_150009.mp4'
counter = 0 # Increment number by 1 at the end of each image so that Images aren't overwritten.
cap = cv2.VideoCapture(video_path)
present = 49 * [False]
success, img = cap.read()
fno = 0
dict = {}
while success:
    if fno % 180 == 0:
        encoded_faces_test = face_recognition.face_encodings(img) # Generate encodings for the faces found in the image.
        currList = []
        for face in encoded_faces_test:
            myList = face_recognition.compare_faces(encoded_face_train, face, tolerance = 0.5)
            for i in range(len(myList)):
                if myList[i]:
                    present[i] = True
                    currList.append(i)
                    imgCopy = img
                    cv2.rectangle(imgCopy, (face[3], face[0]), (face[1], face[2]), (255, 0, 0), 2)
                    cv2.putText(imgCopy, str(i + 1), (face[3], face[0]- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    foundFace = imgCopy[(face[0] - 50):(face[2] + 50), (face[3] - 50):(face[1] + 50)]
                    imgName = str(i + 1) + "-" + str(num) + ".png"
                    cv2.imwrite(os.path.join(r'C:\Users\arsh0\OneDrive\Documents\Smart Attendance\Prayge', imgName), foundFace)
        dict[fno] = currList
    fno += 1
    success, img = cap.read()
print(dict)
for i in range(len(present)):
    if not present[i]:
        print(str(i + 1) + " is not present!")
end = time.time()
print(end - start) # Runtime