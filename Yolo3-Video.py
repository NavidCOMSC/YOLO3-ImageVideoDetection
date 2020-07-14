import numpy as np
import argparse
import imutils
import time
import cv2
import os

#arguments parser to input video file
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="the input directory")
ap.add_argument("-o", "--output", required=True, help="the output directory")
ap.add_argument("-y", "--yolo", required=True, help="directory to Yolo libraries and weights")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="min probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold for applying NMS")
args = vars(ap.parse_args())

#load the pretrained COCO classes and lables on YOLO
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

#initialize a list of colors to associate with labels
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

#access to YOLO weights and configuration file
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

#determine the output layers
print("sanity check to load Yolo files")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#initialization of video stream, frame dimension, pointer to output video
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

#determine total number of frames with throwing exception
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("number of frames {}".format(total))

#if error occurred in determining the number of frames
except:
    print("error has been thrwon to determine the number of frames")
    print("no approximation, the exact time can be printed")
    total = -1

while True: #looping over video frames
    #read and grab frames
    (grabbed, frame) = vs.read()

    #if there is no frame ready to grab, reached end of the file
    if not grabbed:
        break

    #if there is no dimension for frame, provide them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

#construct a blob from input frame and perform a Yolo forward pass
