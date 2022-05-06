import cv2
import pandas as pd
import numpy as np
from datetime import datetime



def detection_visage(img_path, model):
    """Détection de visage sur image"""
    report = pd.read_csv('report.csv')
    tableau = list(report.values)

    # tableau = []

    cascade_path = "./cascades/haarcascade_frontalface_default.xml"

    img_shape = (224, 224)

    src = cv2.cvtColor(np.array(img_path), cv2.COLOR_RGB2BGR)
    cascade = cv2.CascadeClassifier(cascade_path)
    rect = cascade.detectMultiScale(src)

    if len(rect) > 0:
        for i,[x, y, w, h] in enumerate(rect):
            img_trimmed = src[y:y + h, x:x + w]

            # traitement masque
            frame2 = np.array(cv2.resize(img_trimmed, img_shape))/255.0
            frame2 = np.expand_dims(frame2, axis=0)
            test_prob = model.predict(frame2)
            test_pred = test_prob.argmax(axis=-1)
            
            if test_pred == 0:
                txt = 'Mask'
                color = (0, 0, 0)
            else:
                txt = 'No Mask'
                color = (0, 0, 255)

            cv2.rectangle(src, (x, y), (x+w, y+h), color, 3)
            cv2.rectangle(src, (x, y - 30), (x+w, y), color, -1)
            cv2.putText(src, f" Visage {i+1} - {txt}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            date_now = datetime.today()
            tableau.append([f"Visage {i+1}", f"{date_now.day}/{date_now.month}/{date_now.year}", f"{date_now.hour}:{date_now.minute}:{date_now.second}", txt])

    
    tableau = pd.DataFrame(tableau, columns=["Personne", "Date", "Heure", "Statut"])
    tableau.to_csv("report.csv", index=False)

    return cv2.cvtColor(src, cv2.COLOR_BGR2RGB), tableau



def visage_webcam(img, model):
    cascade_path = "./cascades/haarcascade_frontalface_default.xml"

    img_shape = (224, 224)

    cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    rect = cascade.detectMultiScale(cap, scaleFactor=1.1, minNeighbors=2, minSize=(100,100))

    

    if len(rect) > 0:
        for i, [x, y, w, h] in enumerate(rect):
            img_trimmed = cap[y:y + h, x:x + w]

            # traitement masque
            frame2 = np.array(cv2.resize(img_trimmed, img_shape))/255.0
            frame2 = np.expand_dims(frame2, axis=0)
            test_prob = model.predict(frame2)
            test_pred = test_prob.argmax(axis=-1)
            if test_pred == 0:
                txt = 'Mask'
                color = (0, 0, 0)
            else:
                txt = 'No Mask'
                color = (0, 0, 255)
                # color = (0, 255, 0)


            cv2.rectangle(cap, (x, y), (x+w, y+h), color)
            cv2.rectangle(cap, (x, y - 30), (x+w, y), color, -1)
            cv2.putText(cap, f"Visage {i+1} - {txt}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)