import cv2
import mediapipe as mp 
import time
import math


class handDetector():
    def __init__(self , mode = False , maxHands = 2 , model_complexity=1, detectionCon = 0.5 , trackingCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode , self.maxHands , self.model_complexity , self.detectionCon , self.trackingCon)
        self.mpDraw = mp.solutions.drawing_utils
        
    def findHands(self , frame , draw = True):

        imgRGB = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame,handlms, self.mpHands.HAND_CONNECTIONS)
        return frame
    def findPosition(self , frame , handno=0 , draw = True):
        self.lmlist = []
        myHand = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[0]
            for id , lm in enumerate(myHand.landmark):
                    w , h , c = frame.shape
                    cx , cy = int(lm.x*w) , int(lm.y*h)
                    self.lmlist.append([id , cx , cy])
                    
        
        return self.lmlist
    def findFingersUp(self , frame):
        fingersUp = [0 , 0 , 0 , 0 , 0]

        if self.lmlist[8][2] < self.lmlist[6][2]:
            fingersUp[1] = 1
        if self.lmlist[12][2] < self.lmlist[10][2]:
            fingersUp[2] = 1
        if self.lmlist[16][2] < self.lmlist[14][2]:
            fingersUp[3] = 1
        if self.lmlist[20][2] < self.lmlist[18][2]:
            fingersUp[4] = 1
        if self.lmlist[4][1] > self.lmlist[2][1]:
            fingersUp[0] = 1
        
        return fingersUp
    

    def findLength(self ,  x1 , y1 , x2 , y2 , frame , draw = True):
        self.distance = pow(pow((x1 - x2) , 2) + pow(y1-y2, 2) , 0.5)
        return self.distance
        
def main():
    previousTime = 0
    currentTime = 0
    detector = handDetector()

    cap = cv2.VideoCapture(0)
    while True:
            
        _,frame = cap.read() 
        frame = detector.findHands(frame=frame)   
        lmList = detector.findPosition(frame=frame , draw=False)
        if len(lmList) != 0 :
            finguresUp = detector.findFingersUp(frame=frame)
            # print(finguresUp)
            x1 , y1 = lmList[3][1:]
            x2 , y2 = lmList[6][1:]
            length = detector.findLength( x1 , y1 , x2 , y2 ,frame=frame , draw = False )
            print(length)

        currentTime = time.time()
        fps = 1/(currentTime-previousTime)
        previousTime = currentTime
        cv2.putText(frame , str(int(fps)) , (10,70) , cv2.FONT_HERSHEY_COMPLEX , 4 , (255,0,255) , 2)


        if cv2.waitKey(1) == ord('q'):
            break
        cv2.imshow("Frame" , frame)



if __name__ == "__main__":
    main()