import numpy as np
import argparse
import imutils
import time
import cv2
import os
import copy

#arguments parser to input video file
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="the input directory")
ap.add_argument("-o", "--output", required=True, help="the output directory")
ap.add_argument("-r", "--result", required=True, help="the image of person saved directory")
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
print(vs.isOpened())
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
    print("error has been thrown to determine the number of frames")
    print("no approximation, the exact time can be printed")
    total = -1

counter = 0
counterframe = 0
while True: #looping over video frames
    #read and grab frames
    (grabbed, frame) = vs.read()
    #print(len(frame))


    #if there is no frame ready to grab, reached end of the file
    if not grabbed:
        break

    counterframe += 1

    if counterframe % 25 == 0:

        #if there is no dimension for frame, provide them
        if W is None or H is None:
            (H, W) = frame.shape[:2]
            print(frame.shape[:2])

        #deep copying of numpy array
        deepFrame = copy.deepcopy(frame)

        #construct a blob from input frame and perform a Yolo forward pass
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

        #initialization of lists for bounding boxes, confidences and classes
        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            #loop over each of detection
            for detection in output:
                #retrieve the classes and probabilities of detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                #filtering the weak predictions with low probabilities
                if confidence > args["confidence"]:
                    # scaling the bounding box coordinates wrt the center of box returned from Yolo
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    #use the center coordiantes to compute box corners
                    x = int(centerX - (width / 2)) #left
                    y = int(centerY - (height / 2)) #top

                    #populate the lists of bounding boxes, confidences and classes
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    #print(classIDs)


        #Apply the non-maximun suppression to remove the overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
        #check for the existence of detections:
        if len(idxs) > 0:
            for i in idxs.flatten():

                #Bounding Boxes corrdination and geometry
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                if LABELS[classIDs[i]] == "person":

                    #deepFrame = copy.deepcopy(frame)
                    person_img = deepFrame[y:y+h, x:x+w, :]
                    cv2.imwrite(os.path.sep.join([args["result"], f'ShopFrontLeft{counter}.png']), person_img)
                    #cv2.imwrite(os.path.sep.join([args["result"], "person{}.png".format(counter)]), person_img)
                    counter = counter + 1


                #draw a bounding box rectangle and label on the frame
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        #check if the video writer is None
        if writer is None:
            #initialize the video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

            #processing time on single frame
            if total > 0:
                elap = (end - start)
                print("single frame took {:.4f} seconds".format(elap))
                print("estimated total time to process: {:.4f}".format(elap * total))

        #write the output frame to disk
        writer.write(frame)

#release and clean up the file pointers
writer.release()
vs.release()

