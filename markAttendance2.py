from helper import *

images = []
classLabels = []
load_data(images, classLabels)
savePath = r'C:\Users\arsh0\OneDrive\Documents\Smart Attendance\Prayge' #Path to IMG with Label and a random garbage value
currPath = r'C:\Users\arsh0\OneDrive\Documents\Smart Attendance' #Working directory.

encoded_face_train = []
#encoded_face_train = generateEncodings(images)
#Uncomment the line above if using first time. Re-comment after.
save_encoding(encoded_face_train)  
encoded_face_train = load_encoding()

start = time.time() #To calculate Runtime.
path = r'C:\Users\arsh0\OneDrive\Documents\Smart Attendance\Face Detected' #Faces detected using YOLO SSD.
counter = 0 # Increment number by 1 at the end of each image so that Images aren't overwritten.
for imgName in os.listdir(path):
    img = cv2.imread(os.path.join(path, imgName))
    faceInImg = [(0, img.shape[0], img.shape[1], 0)]
    y1, x2, y2, x1 = faceInImg[0] # top, right, bottom, left coords of faces.
    encoded_faces_test = face_recognition.face_encodings(img, faceInImg) # Generate encodings for the faces found in the image.

    for i in range(len(encoded_faces_test)):
        faceDist = face_recognition.face_distance(encoded_face_train, encoded_faces_test[i]) # Calculate Euclidean distance between face in image, actual face.
        minMatch = np.argmin(faceDist)
        if(faceDist[minMatch] < 0.45): # IF a close enough match is found Attendance is marked.
            print(str(classLabels[minMatch]) + " is found!")
            os.chdir(savePath)
            cv2.imwrite(str(classLabels[minMatch] + "-" + str(counter)) + ".jpg",img) # Name Format: "Label-GarbageCounter.jpg"
            os.chdir(currPath)
            counter += 1
end = time.time()
print(end - start) # Runtime