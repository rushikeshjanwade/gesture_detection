import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow

cap = cv2.VideoCapture(0)  # Device id is 0
detector=HandDetector(maxHands=1)
classifier=Classifier("model/keras_model.h5","model/labels.txt")


offset=20
imgsize=300


while True:
    sucess, img = cap.read()
    hands,img=detector.findHands(img)
    if hands:
        lhand=hands[0]
        x,y,w,h=lhand["bbox"]

        imgwhite=np.ones((imgsize,imgsize,3),np.uint8)*255
        imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgcropshape=imgcrop.shape

        aspectratio=h/w

        if aspectratio>1:
            k=imgsize/h
            wcalc=math.ceil(k*w)
            imgresize=cv2.resize(imgcrop,(wcalc,imgsize))
            imgresizeshape=imgresize.shape
            wGap=math.ceil((300-wcalc)/2)
            imgwhite[:,wGap:wcalc+wGap]=imgresize

            prediction,index=classifier.getPrediction(imgwhite)
            # prediction, index = classifier.getPrediction(imgwhite)
            print(prediction,index)


        else:
            k = imgsize / w
            hcalc = math.ceil(k * h)
            imgresize = cv2.resize(imgcrop, (imgsize, hcalc))
            imgresizeshape = imgresize.shape
            hGap = math.ceil((300 - hcalc) / 2)
            imgwhite[hGap:hcalc+hGap,:] = imgresize

            prediction, index = classifier.getPrediction(imgwhite)
            # prediction, index = classifier.getPrediction(imgwhite)
            print(prediction, index)

        cv2.imshow("Imagewhite", imgwhite)
        # cv2.imshow("Imagecrop", imgcrop)

    cv2.imshow("Image", img)
    key=cv2.waitKey(1)
