from imageai.Detection import ObjectDetection
import os
import cv2
import time

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

i = 0
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"))
img = cv2.imread("imagenew.jpg")
cv2.imshow('frame', img)

for eachObject in detections:
    print(eachObject)
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
