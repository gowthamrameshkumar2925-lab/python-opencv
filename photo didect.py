import cv2 as cv
trainedDataset=cv.CascadeClassifier("E:/Gowtham/haarcascade_frontalface_default.xml")

img=cv.imread("E:/Gowtham/opencv/vijay.png")
gray=cv.cvtColor(img,cv.COLOR_BGR2RGB)

faces=trainedDataset.detectMultiScale(gray)
print(faces)
for x,y,w,h in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv.rectangle(gray,(x,y),(x+w,y+h),(255,0,0),2)

cv.imshow("anna",img)
cv.imshow("anna2",gray) 
cv.waitKey()
