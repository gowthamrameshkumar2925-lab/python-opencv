import cv2 as cv

trainedDataset=cv.CascadeClassifier("E:/Gowtham/haarcascade_frontalface_default.xml")

video=cv.VideoCapture(0)
while True:
    success,frame=video.read()
    if success==True:
        gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces=trainedDataset.detectMultiScale(gray)
        print(faces)
        for x,y,w,h in faces:
            cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv.imshow("video",frame)
        cv.waitKey(1)
else:
    print("The video end")
    
