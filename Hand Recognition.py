import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)  

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:  #this line check didect hands
        for hand_landmarks in results.multi_hand_landmarks:  #Loops through each detected hand
            for lm in hand_landmarks.landmark:    #landmark means 21 hand points
                height, width, channel = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Video", img)

    if (cv2.waitKey(1) & 0xFF==ord("q")) :  
        break

cap.release()
cv2.destroyAllWindows()
