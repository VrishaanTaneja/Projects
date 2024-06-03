import mediapipe as mp
import cv2
import time


cap = cv2.VideoCapture(0)
try:
    mpHands=mp.solutions.hands
    hands=mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils
    ptime=0
    ctime=0
    while True:
        success, img = cap.read()
        #cv2.imshow("Image", img)
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        result = hands.process(imgRGB)
        #print(result.multi_hand_landmarks)
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h,w,c=img.shape
                    cx,cy=int(lm.x*w),int(lm.y*h)
                    print(id,cx,cy)


                    #if id == 4 or id == 8:
                        #cv2.circle(img,(cx,cy),10,(255,0,0),cv2.FILLED)
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255),1)
 
        cv2.imshow('Image',img)
        cv2.waitKey(1)

except KeyboardInterrupt:
    print("image exited by user")