import cv2
import time
import HandTrackingModule as htm
import random
cap = cv2.VideoCapture(0)
detector=htm.HandDetector(maxHands=1, detectionCon=0.8)
ptime=0

overlaylist=[1,2,3,4,5,6,7,8,9,10,20]

try:
    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tip_id=[0,1,3,4,5,6,8,10,12,14,16,18,20]
        lst=detector.retlmlst_hands(imgRGB,tip_id)
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime

        print(lst)
        if len(lst)!=0:
            finger = []
            if (lst[0][1]<=lst[1][1]):
                if (lst[3][1]>=lst[4][1]):
                    finger.append(1)
                else:
                    finger.append(0)
            else:
                if (lst[3][1]<=lst[4][1]):
                    finger.append(1)
                else:
                    finger.append(0)
            i = 6
            while i < len(lst):
                if i < len(lst):
                    if lst[i][2] < lst[i - 1][2]:
                        finger.append(1)
                        i += 2
                    else:
                        finger.append(0)
                        i += 2
            print(finger)
            sumf=0
            for i in finger:
                sumf+=i
            cv2.putText(img,'count:'+ str(sumf),(10,70),cv2.FONT_HERSHEY_PLAIN,1,(255,0,255))
        cv2.imshow('img',img)
        cv2.waitKey(1)

except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()