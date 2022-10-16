from dependencies import *
def generateEncodings(images):
    '''
        Takes a list of images as input and generates a list of Encodings for faces in images.
        Params images -> List of Images
        Output: encodedList -> List of Facial Encodings 
    '''
    encodeList = []
    for img in images:
        token = face_recognition.face_encodings(img, num_jitters=10)[0]
        encodeList.append(token)
    return encodeList

def load_data(images, classLabels):
    '''
        Takes 2 empty Lists and generates images, ClassLabels corresponding to the images
        images -> contains images
        classLabels -> corresponding Labels
    '''
    path = r'C:\Users\arsh0\OneDrive\Documents\Smart Attendance\IAmNotDoingThisAgain' #Path to Train Face Encoding Images
    for img in os.listdir(path):
        trainImg = cv2.imread(os.path.join(path, img))
        images.append(trainImg)
        classLabels.append(img[0:2])

def save_encoding(myList):
    '''
        Ensures that you only have to generate Encodings once. Comment out generate_encodings if used before.
    '''
    with open("test", "wb") as fp:
        pickle.dump(myList, fp)

def load_encoding():
    '''
        Loads Saved encodings for all registered students in a class.
        returns encodingList -> List of all Encodings
    '''
    encodingList = []
    with open("test", "rb") as fp:
        encodingList = pickle.load(fp)
    return encodingList
