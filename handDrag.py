import cv2
import numpy as np
import handTrackingModule as htm
import time
import pyautogui

##########################
wCam, hCam = 640, 480
frameR = 100  
smoothening = 5
scroll_threshold = 10 
click_threshold = 40   
drag_threshold = 50   
#########################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
scrollY = 0
dragging = False  # State to check if dragging is active

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = pyautogui.size()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        x_thumb, y_thumb = lmList[4][1:]

        #Check which fingers are up
        fingers = detector.findFingersUp(img)

        #Convert coordinates for cursor movement
        x_mapped = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
        y_mapped = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

        clocX = plocX + (x_mapped - plocX) / smoothening
        clocY = plocY + (y_mapped - plocY) / smoothening

        #Pinch Gesture: Thumb and Index Finger close together for Dragging
        distance_thumb_index = np.hypot(x_thumb - x1, y_thumb - y1)  # Distance between thumb and index finger

        if distance_thumb_index < drag_threshold:
            if not dragging:
                pyautogui.mouseDown(button='left')  # Simulate pressing the left mouse button
                dragging = True
            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.putText(img, "Dragging", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
            cv2.circle(img, (int(x1), int(y1)), 15, (0, 255, 255), cv2.FILLED)

        #Release the drag when pinch is released
        elif dragging:
            pyautogui.mouseUp(button='left')  # Simulate releasing the left mouse button
            dragging = False
            cv2.putText(img, "Released", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        #Only index finger: Moving Mode
        elif fingers[1] == 1 and fingers[2] == 0:
            pyautogui.moveTo(wScr - clocX, clocY)
            cv2.circle(img, (int(x1), int(y1)), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        #Three fingers up: Scroll Mode
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            scroll_speed = y1 - scrollY

            if abs(scroll_speed) > scroll_threshold:
                pyautogui.scroll(int(-scroll_speed / 2))  
                scrollY = y1

            cv2.putText(img, "Scrolling", (10, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        #Index and Middle Finger close together: Click Mode
        elif fingers[1] == 1 and fingers[2] == 1:
            distance = np.hypot(x2 - x1, y2 - y1) 

            if distance < click_threshold:
                pyautogui.click()
                cv2.circle(img, (int(x1), int(y1)), 15, (0, 255, 0), cv2.FILLED)
                cv2.putText(img, "Click", (10, 130), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
