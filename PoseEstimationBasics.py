import cv2
import mediapipe as mp
import time
import HandTrackingModule

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

detector= HandTrackingModule.HandDetector()
cap = cv2.VideoCapture(0)
ptime=0
try:
    while True:
        success, img = cap.read()


        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)
        print(results.pose_landmarks)
        if results.pose_landmarks:
            mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
            for id,lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                print(id,lm)
                cx,cy=int(lm.x*w),int(lm.y*h)
                cv2.circle(img,(cx,cy),8,(255,0,0),cv2.FILLED)
        imgRGV = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()