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
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

#Path for loading Yolo3 weights and configuration files
weightpath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configpath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

#loading YOLO object detector trained on COCO dataset (80 features)
print("loading YOLO from disk")
net = cv2.dnn.readNetFromDarknet(configpath, weightpath)

#Load the input image and extract the image dimension
image  = cv2.imread(args["image"])
(H, W) = image.shape[:2]

#determine the output layer
lay = net.getLayerNames()
lay = [lay[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#Construct the blob from image and peform a forward pass
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(lay)
end = time.time()

print("YOLO on MAC takes {:.6f} seconds".format(end - start))

#initialization of lists for bounding boxes, confidences and class IDs
boxes = []
confidences = []
classIDs = []

# loop over each of layer outputs
for output in layerOutputs:
    for detection in output:
        # extract the class ID and probability of each class
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]

        # Apply the confidence threshold
        if confidence > args["confidence"]:
            # scaling the bounding box with respect to the cneter of boxes returned by YOLO
            box = detection[0:4] * np.array([W, H, W, H])
            (centerX, centerY, width, height) = box.astype("int")

            # derive the width and height of box
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            #populating the lists of coordinates, confidence and class IDs
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(classID)

#apply non-maxima suppression to suppress weak, overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

#sanity check for the existence of element in the detection
if len(idxs) > 0:
    #loop over indexes
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        #draw a bounding box with colour and label on them
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# display output image
cv2.imshow("Image", image)
cv2.waitKey(0)