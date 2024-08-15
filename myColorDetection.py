import cv2 
import numpy as np
from PIL import Image
import math

cap = cv2.VideoCapture(0)
imgPicker = np.zeros((200,400,3), np.uint8)

width = 3448  # Larghezza desiderata
height = 808 # Altezza desiderata
cutGreen = 48
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

fps = 60  # FPS desiderati
cap.set(cv2.CAP_PROP_FPS, fps)

actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
actual_fps = cap.get(cv2.CAP_PROP_FPS)

height_cut = 800
width_cut = 1328

print(f"Risoluzione impostata: {width}x{height}")
print(f"FPS impostati: {fps}")
print(f"Risoluzione effettiva: {actual_width}x{actual_height}")
print(f"FPS effettivi: {actual_fps}")

global callbackable
callbackable = True

colorBGRMin = np.array([[[0,0,0]]], np.uint8) #creo un colore come se fosse un' immagine da un pixel con 3 canali
colorBGRMax = np.array([[[0,0,0]]], np.uint8) #uguale a sopra ma per hsv
colorHSVMin = np.array([[[0,0,0]]], np.uint8)
colorHSVMax = np.array([[[0,0,0]]], np.uint8)  

points = []
speeds = []
scarto = 5
pixelSize = 0.00026 # m/px

def changeHSVMin(x):

    global callbackable
    if not callbackable:
        return 1
    
    h = cv2.getTrackbarPos('H_MIN', 'color picker')
    s = cv2.getTrackbarPos('S_MIN', 'color picker')
    v = cv2.getTrackbarPos('V_MIN', 'color picker')

    colorHSVMin[:] = [[h, s, v]]

    BGRColor = cv2.cvtColor(colorHSVMin, cv2.COLOR_HSV2BGR)

    callbackable = False
    cv2.setTrackbarPos('R_MIN', 'color picker', BGRColor[0][0][2])
    cv2.setTrackbarPos('G_MIN', 'color picker', BGRColor[0][0][1])
    cv2.setTrackbarPos('B_MIN', 'color picker', BGRColor[0][0][0])

    r = cv2.getTrackbarPos('R_MIN', 'color picker')
    g = cv2.getTrackbarPos('G_MIN', 'color picker')
    b = cv2.getTrackbarPos('B_MIN', 'color picker')

    cv2.rectangle(imgPicker, (0,0), (200,200), (b, g, r), -1)
    cv2.putText(imgPicker, 'min', (0,25), 0, 1, (0, 0, 0), 2, )
    callbackable = True

    return 0


def changeBGRMin(y):

    global callbackable
    if not callbackable:
        return 1
    
    r = cv2.getTrackbarPos('R_MIN', 'color picker')
    g = cv2.getTrackbarPos('G_MIN', 'color picker')
    b = cv2.getTrackbarPos('B_MIN', 'color picker')

    colorBGRMin[:] = [[b, g, r]]
    HSVColor = cv2.cvtColor(colorBGRMin, cv2.COLOR_BGR2HSV) #converto il colore come se fosse un pixel di un' immagine con 3 canali
    callbackable = False
    cv2.setTrackbarPos('H_MIN', 'color picker', HSVColor[0][0][0])
    cv2.setTrackbarPos('S_MIN', 'color picker', HSVColor[0][0][1])
    cv2.setTrackbarPos('V_MIN', 'color picker', HSVColor[0][0][2])
    cv2.rectangle(imgPicker, (0,0), (200,200), (b,g,r),-1)
    cv2.putText(imgPicker, 'min', (0,25), 0, 1, (0, 0, 0), 2, )
    callbackable = True

    return 0

def changeBGRMax(z):

    global callbackable
    if not callbackable:
        return 1
    
    r = cv2.getTrackbarPos('R_MAX', 'color picker')
    g = cv2.getTrackbarPos('G_MAX', 'color picker')
    b = cv2.getTrackbarPos('B_MAX', 'color picker')

    colorBGRMax[:] = [[b, g, r]]
    HSVColor = cv2.cvtColor(colorBGRMax, cv2.COLOR_BGR2HSV) #converto il colore come se fosse un pixel di un' immagine con 3 canali
    callbackable = False
    cv2.setTrackbarPos('H_MAX', 'color picker', HSVColor[0][0][0])
    cv2.setTrackbarPos('S_MAX', 'color picker', HSVColor[0][0][1])
    cv2.setTrackbarPos('V_MAX', 'color picker', HSVColor[0][0][2])
    cv2.rectangle(imgPicker, (200,0), (400,200), (b,g,r),-1)
    cv2.putText(imgPicker, 'max', (330,25), 0, 1, (0, 0, 0), 2, )
    callbackable = True

    return 0

def changeHSVMax(q):

    global callbackable
    if not callbackable:
        return 1
    
    h = cv2.getTrackbarPos('H_MAX', 'color picker')
    s = cv2.getTrackbarPos('S_MAX', 'color picker')
    v = cv2.getTrackbarPos('V_MAX', 'color picker')

    colorHSVMax[:] = [[h, s, v]]

    BGRColor = cv2.cvtColor(colorHSVMax, cv2.COLOR_HSV2BGR)

    callbackable = False
    cv2.setTrackbarPos('R_MAX', 'color picker', BGRColor[0][0][2])
    cv2.setTrackbarPos('G_MAX', 'color picker', BGRColor[0][0][1])
    cv2.setTrackbarPos('B_MAX', 'color picker', BGRColor[0][0][0])

    r = cv2.getTrackbarPos('R_MAX', 'color picker')
    g = cv2.getTrackbarPos('G_MAX', 'color picker')
    b = cv2.getTrackbarPos('B_MAX', 'color picker')

    cv2.rectangle(imgPicker, (200,0), (400,200), (b, g, r), -1)
    cv2.putText(imgPicker, 'max', (330,25), 0, 1, (0, 0, 0), 2, )
    callbackable = True

    return 0

def colorPickerTrackbar():

    cv2.namedWindow('color picker') 
    cv2.createTrackbar('R_MIN', 'color picker', 0, 255, changeBGRMin)
    cv2.createTrackbar('G_MIN', 'color picker', 0, 255, changeBGRMin)
    cv2.createTrackbar('B_MIN', 'color picker', 0, 255, changeBGRMin)
    cv2.createTrackbar('R_MAX', 'color picker', 0, 255, changeBGRMax)
    cv2.createTrackbar('G_MAX', 'color picker', 0, 255, changeBGRMax)
    cv2.createTrackbar('B_MAX', 'color picker', 0, 255, changeBGRMax)
    cv2.createTrackbar('H_MIN', 'color picker', 0, 179, changeHSVMin)
    cv2.createTrackbar('S_MIN', 'color picker', 0, 255, changeHSVMin)
    cv2.createTrackbar('V_MIN', 'color picker', 0, 255, changeHSVMin)
    cv2.createTrackbar('H_MAX', 'color picker', 0, 179, changeHSVMax)
    cv2.createTrackbar('S_MAX', 'color picker', 0, 255, changeHSVMax)
    cv2.createTrackbar('V_MAX', 'color picker', 0, 255, changeHSVMax)
    cv2.resizeWindow('color picker', 400, 400)

    return 0


def calculateSpeed(center_x, center_y, oldCenter_x, frame): #da guardare

    avgSpeed = 0

    if(center_x < oldCenter_x + scarto):

        avgSpeed = 0
        points.clear()
        speeds.clear()
    else:
        points.append((center_x, center_y))
    
    oldCenter_x = center_x
    
    for i in range(len(points)):

        speed = math.sqrt((points[i][0] - points[i-1][0]) ** 2 + (points[i][1] - points[i-1][1]) ** 2)
        speeds.append(speed)


    for speed in speeds:
        avgSpeed += speed
    
    if len(speeds) != 0:

        avgSpeed = avgSpeed / len(speeds)
        avgSpeed = avgSpeed * fps

    if avgSpeed != 0:
       print(f"speed = {avgSpeed} px/s")
       cv2.putText(frame, f'speed = {round(avgSpeed, 2)} px/s', (20,30), 0, 1, (255, 0, 0), 2, )
    else:
       cv2.putText(frame, f'speed = 0 px/s', (20,30), 0, 1, (255, 0, 0), 2, )
    
    return 0

def predictTrajectory(center_x, center_y, oldCenter_x, frame): #trendline source: https://classroom.synonym.com/calculate-trendline-2709.html
    
    goal_x, goal_y = None, None
    
    if(center_x < oldCenter_x + scarto):

        points.clear()
    else:

        points.append((center_x, center_y))
    
    oldCenter_x = center_x

    if len(points) > 1:

        a = 0
        sum_x = 0
        sum_y = 0
        sumSquare_x = 0

        #n:
        n = len(points)

        for point in points:

            a += point[0] * point[1]
            sum_x += point[0]
            sum_y += point[1]
            sumSquare_x += point[0] ** 2
        
        a *= n
        b = sum_x * sum_y
        c = n * sumSquare_x
        d = sum_x ** 2
        
        m = (a - b) / (c - d) if (c - d) != 0 else 0 #pendenza --> ricordarsi che il sistema non rileva punti in verticale
        x = width
        y = int(m * (x - center_x) + center_y) #considerare che si lavora in (x→+, y↓+)
        q = (center_y - m * center_x)

        cv2.line(frame, (0, int(q)), (x, y), ( 0, 0, 255), 5)
    
        if 0 <= y <= height:
            rebounding = False
        else:
            rebounding = True

        firstHit = True

        while rebounding:

            if firstHit == True:

                if m < 0:
                    bounce_x = int(-q / m)
                    bounce_y = 0
                else: # m>0
                    bounce_x = int((height - q) / m)
                    bounce_y = height
                    cv2.line(frame, (center_x, center_y), (bounce_x, bounce_y), (0, 255, 0), 5)

                firstHit = False

            m *= -1
            q = (bounce_y - m * bounce_x)

            if m < 0:
                bounce_x1 = int(-q / m)
                bounce_y1 = 0
            else: # m>0
                bounce_x1 = int((height - q) / m)
                bounce_y1 = height

            if bounce_x1 >= width: #bounce_xy1 diventano il goal
                bounce_x1 = width
                bounce_y1 = int((m * bounce_x1 + q))
                rebounding = False

            try:
                cv2.line(frame, (bounce_x, bounce_y), (bounce_x1, bounce_y1), (255, 0, 0), 5)
            except UnboundLocalError:

                print("Unbound Local Error: passed to the next iteration")
                pass
            
            bounce_x, bounce_y = bounce_x1, bounce_y1

        else: #disegna la linea principale
            cv2.line(frame, (center_x, center_y), (x, y), (0, 255, 0), 5)

        try:
            print(f' goal at ({bounce_x1}, {bounce_y1})') if bounce_x1 != None and bounce_y1 != None else (x, y)
            return bounce_x1, bounce_y1 if bounce_x1 != None and bounce_y1 != None else (x, y)
        except UnboundLocalError:
            print("Unbound Local Error: passed to the next iteration")
            pass
    
    return 1

def draw_points(frame):
    for point in points:
        cv2.circle(frame, point, 5, (0, 0, 255), -1)


def main():

    colorPickerTrackbar()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("errore lettura frame")
            break

        #taglia il frame
        frame = frame[0:height_cut, cutGreen:width_cut] #adattamento alla camera sony
        
        #maschera
        lowerLimit = np.array([cv2.getTrackbarPos('H_MIN', 'color picker'), 
                    cv2.getTrackbarPos('S_MIN', 'color picker'), 
                    cv2.getTrackbarPos('V_MIN', 'color picker')])
        upperLimit = np.array([cv2.getTrackbarPos('H_MAX', 'color picker'), 
                    cv2.getTrackbarPos('S_MAX', 'color picker'), 
                    cv2.getTrackbarPos('V_MAX', 'color picker')])
        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
        result = cv2.bitwise_and(frame, frame, mask=mask) #mostra solo i frame della maschera

        mask_ = Image.fromarray(mask)
        bbox = mask_.getbbox()
        #print(bbox)

        if bbox is not None:
            x1, y1, x2, y2 = bbox

            center_x, center_y = (x2 + x1) // 2, (y2 + y1) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(frame, (center_x, center_y), 10, (255, 0, 0), -1)

        if len(points) == 0: 
            oldCenter_x = 0

        #calculateSpeed(center_x, center_y, oldCenter_x, frame)
        predictTrajectory(center_x, center_y, oldCenter_x, frame)
        draw_points(frame)

        oldCenter_x = center_x


        cv2.imshow('result', result)
        cv2.imshow('Frame', frame)
        cv2.imshow('color picker', imgPicker)

        
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#######################################################################################################
#                                                                                                     #
#                                                                                                     #
# LAM Robotica 2024-2025                                                                              #
#                                                                                                     #                                                                                           #
# Last edit: 13.8.2024 - 16:40                                                                        #
#                                                                                                     #
# @fd                                                                                                 #
#                                                                                                     #
#                                                                                                     #
#######################################################################################################