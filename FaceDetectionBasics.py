import mediapipe as mp
import cv2
import time

cap = cv2.VideoCapture(0)
ptime=0;

mpface=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
faceDetect=mpface.FaceDetection()

try:
    while True:
        success,img= cap.read()
        imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results=faceDetect.process(imgRGB)
        if results.detections:
            for id,detection in enumerate(results.detections):
                #print(id,detection)
                #print(detection.score)
                #print(detection.location_data.relative_bounding_box)
                mpDraw.draw_detection(img,detection)
                bboxC=detection.location_data.relative_bounding_box
                ih,iw,ic = img.shape
                bbox = int(bboxC.xmin * iw),int(bboxC.ymin * ih), \
                       int(bboxC.width * iw),int(bboxC.height * ih)
                cv2.rectangle(img,bbox,(255,0,0),2)
                cv2.putText(img, f':{int(detection.score[0]*100)}%', (bbox[0],bbox[1]-25), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        ctime =time.time()
        fps=1/(ctime-ptime)
        ptime=ctime

        cv2.putText(img,f'FPS:{int(fps)}',(10,40),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
        cv2.imshow("Image",img)
        cv2.waitKey(1)
except KeyboardInterrupt:
    cap.release()
    cv2.destroyAllWindows()
    print('Exiting Program')
