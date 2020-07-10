import numpy as np
import argparse
import time
import cv2
import os

""""
This file first parses the input directory to fetch an image and using YoloV3
to detect objects and draw bounding boxes around objects
"""

#Parse the necassary arguments for image and Yolo3
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required= True, help="input directory to load an image")
ap.add_argument("-y", "--yolo", required= True, help= "path to the YOLO directory")
ap.add_argument("-C", "--confidence", type=float, default=0.5, help="min prob ti filer weak detection")
ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())

#COCO feature labels that YOLO is pretrained with
labelspath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelspath).read().strip().split("\n")

#randon color initialization
np.random.seed(42)
COLORS = np.randon.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

