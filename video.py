import cv2 as cv

trainedDataset=cv.CascadeClassifier("E:/Gowtham/haarcascade_frontalface_default.xml")

video=cv.VideoCapture("E:\Gowtham\Opencv\Cadbury Gems _ Telly.mp4")
while True:
    success,frame=video.read()
    cv.imshow("video",frame)
    cv.waitKey(1)
