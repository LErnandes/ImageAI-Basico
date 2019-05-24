from imageai.Detection import VideoObjectDetection
import os
import cv2
import numpy as np

execution_path = os.getcwd()

camera = cv2.VideoCapture(0)

def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(camera_input=camera, save_detected_video = False,
    per_frame_function=forFrame, minimum_percentage_probability=30)
