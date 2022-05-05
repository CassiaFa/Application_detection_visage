import cv2
import pandas as pd
import numpy as np
from datetime import datetime



def detection_visage(img_path):

    tableau = []

    cascade_path = "./cascades/haarcascade_frontalface_default.xml"
    color = (0, 0, 0)

    src = cv2.cvtColor(np.array(img_path), cv2.COLOR_RGB2BGR)
    cascade = cv2.CascadeClassifier(cascade_path)
    rect = cascade.detectMultiScale(src)

    if len(rect) > 0:
        for i,[x, y, w, h] in enumerate(rect):
            cv2.rectangle(src, (x, y), (x+w, y+h), color, 3)
            cv2.rectangle(src, (x, y - 30), (x+w, y), color, -1)
            cv2.putText(src, f"Visage {i+1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            date_now = datetime.today()
            tableau.append([f"Visage {i+1}", f"{date_now.day}/{date_now.month}/{date_now.year}", f"{date_now.hour}:{date_now.minute}:{date_now.second}"])

    
    tableau = pd.DataFrame(tableau, columns=["Personne", "Date", "Heure"])

    return cv2.cvtColor(src, cv2.COLOR_BGR2RGB), tableau



def visage_webcam(cap):
    cascade_path = "./cascades/haarcascade_frontalface_default.xml"

    cascade = cv2.CascadeClassifier(cascade_path)

    rect = cascade.detectMultiScale(cap, scaleFactor=1.1, minNeighbors=2, minSize=(30,30))

    color = (0, 0, 0)

    if len(rect) > 0:
        for i, [x, y, w, h] in enumerate(rect):
            cv2.rectangle(cap, (x, y), (x+w, y+h), color)
            cv2.rectangle(cap, (x, y - 30), (x+w, y), color, -1)
            cv2.putText(cap, f"Visage {i+1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    return cap