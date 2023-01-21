from yoloface import face_analysis
import cv2
import os
face = face_analysis()
fileNum = 1
cap = cv2.VideoCapture(r'C:\Users\arsh0\OneDrive\Documents\Smart Attendance\VID_20221102_150009.mp4')
_ = True
count = 0
while _: 
    _, frame = cap.read()
    if(count % 20 != 0):
        count += 1
        continue
    if(_ == False):
        break
    #frame = cv2.rotate(frame, cv2.ROTATE_180)
    __,boxes,conf=face.face_detection(frame_arr=frame,frame_status=True,model='full')
    output_frame=face.show_output(img=frame,face_box=boxes,frame_status=True)
    for box in boxes:
        x,y,w,h = box
        outputImg = output_frame[y:y+h, x:x+w]  
        if(outputImg.shape[1] == 0):
            continue
        imageFileName = "outputImg" + str(count) + ".jpg"
        cv2.imwrite(os.path.join(r'C:\Users\arsh0\OneDrive\Documents\Smart Attendance\Face Detected', imageFileName), outputImg)
        count += 1
    # window_name = 'image'
    # img = Image.fromarray(__)
    # img = Image.fromarray(output_frame)
    # fileName = str(fileNum) + ".jpg"
    # fileNum += 1
    # img.save(os.path.join(r"C:\Users\arsh0\OneDrive\Documents\Smart Attendance\FrameByFrame", fileName))
cap.release()
cv2.destroyAllWindows()