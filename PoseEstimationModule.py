import cv2
import mediapipe as mp
import time

class PoseEstimation():
    def __init__(self):
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose()

    def findpose(self,img,imgRGB,draw=False):
        results = self.pose.process(imgRGB)
        #print(results.pose_landmarks)
        if results.pose_landmarks:
            self.mpDraw.draw_landmarks(img, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                if draw == True:
                    cv2.circle(img, (cx, cy), 8, (255, 0, 0), cv2.FILLED)
        return(img)

    def getpos(self,imgRGB,lst):
        results = self.pose.process(imgRGB)
        # print(results.pose_landmarks)
        if results.pose_landmarks:
            self.mpDraw.draw_landmarks(imgRGB, results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = imgRGB.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lst.append([id,cx,cy])
        return(lst)

#dummy code
def main():
    cap = cv2.VideoCapture(0)
    ptime=0
    detector = PoseEstimation()
    lst=[]
    try:
        while True:
            success, img = cap.read()
            imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=detector.findpose(img,imgRGB,draw=True)

            detector.getpos(imgRGB,lst)

            ctime = time.time()
            fps = 1 / (ctime - ptime)
            ptime = ctime
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)
            cv2.imshow('Image', img)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()
        print(lst)



if __name__=="__main__":
    main()