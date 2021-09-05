# Modified https://github.com/yashs97/object_tracker/blob/master/multi_label_tracking.py

import numpy as np
import argparse
import cv2 
import time
from imutils.video import FPS 
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import torchvision

import random

import operator

# TODO: run detection every Nth frame and take the frame with the highest number of detections.
# Can we know that?
# How to connect current detections with previous detection results?

# construct the argument parse 
parser = argparse.ArgumentParser(description='Script to run Object trackers using opencv')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--thr", default=0.6, type=float, help="confidence threshold to filter out weak detections")
parser.add_argument("--frame_count", default='5',help="run the object detector every n frames")
parser.add_argument("--output",default = False,help = "create output video file")
parser.add_argument("-m", "--model", default="./output/faster-rcnn-corn_bgr8_ep100.pt",
                help="path to the model")
args = parser.parse_args()

CLASS_NAMES = ["__background__", "corn_stem"]
def get_prediction(img, confidence=0.5):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold value for prediction score
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.
    
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # model = torch.load(args["model"])
    model = torch.load("./output/faster-rcnn-corn_bgr8_ep100.pt")
    # img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img).to(device)
    pred = model([img])
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if len([x for x in pred_score if x>confidence])!=0:
      pred_t = [pred_score.index(x) for x in pred_score if x>confidence][-1]
      pred_boxes = pred_boxes[:pred_t+1]
      pred_class = pred_class[:pred_t+1]
      pred_score = pred_score[:pred_t+1]
    else:
      pred_boxes, pred_class, pred_score = None, None, None

    return pred_boxes, pred_class, pred_score

# Labels of Network.
labels = { 0: 'background', 1: 'corn'}

lk_params = dict(winSize = (50,50), maxLevel = 4, 
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Open video file or capture device. 
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)

fps = FPS().start()
total_frames = 1
_, prev_frame = cap.read()
tracking_started = False

if args.output:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(args.output, fourcc, 100,(prev_frame.shape[1], prev_frame.shape[0]), True)

# color_dict = dict()

bbox_dict = dict()
cnt = 0 ##########
while True:
    _,frame = cap.read()
    if frame is None: #end of video file
        break
    frame_resized = cv2.resize(frame,(300,300)) # reshaping frame to (300,300)
    # running the object detector every nth frame 
    if total_frames % int(args.frame_count)-1 == 0:
        
        pred_boxes, pred_class, pred_score = get_prediction(frame, 0.5)
        
        centroids = np.zeros([1, 1, 2], dtype=np.float32)

        # only if there are predictions
        if pred_boxes != None:
            corn_dict = dict()
            for i in range(len(pred_boxes)):
                corn_dict[i]=dict()
            corn_dict['centroids']=dict()

            for i in range(len(pred_boxes)):
            # cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=rect_th)
                color = list(np.random.random(size=3) * 256)
                # print("i color", i, color)
                tracking_id = int(i)
                confidence = pred_score[i]

                xLeftBottom = int(pred_boxes[i][0][0]) 
                yLeftBottom = int(pred_boxes[i][0][1])
                xRightTop   = int(pred_boxes[i][1][0])
                yRightTop   = int(pred_boxes[i][1][1])

                # print class and confidence          
                label = pred_class[i] +": "+ str(confidence)             
                # print(label)

                x = (xLeftBottom + xRightTop)/2
                y = (yLeftBottom + yRightTop)/2

                corn_dict[i]['bbox'] = [(xLeftBottom,yLeftBottom),(xRightTop,yRightTop)]
                corn_dict[i]['centroid'] =[(x,y)]
                corn_dict['centroids'][tuple((x,y))]=[]

                # bbox_dict[tuple((x,y))]=[(xLeftBottom,yLeftBottom),(xRightTop,yRightTop)]
                # print("bbox_dict", bbox_dict)
                frame = cv2.rectangle(frame,(xLeftBottom,yLeftBottom),(xRightTop,yRightTop), color, thickness=2) ### added today
                # draw the centroid on the frame
                frame = cv2.circle(frame, (int(x),int(y)), 15, color, -1)
                print("before if STATE i %d frame %d x y: %d %d" % (i, total_frames, x, y))
                tracking_started = True
                if i == 0:
                    color_dict = dict()
                    centroids[0,0,0] = x
                    centroids[0,0,1] = y
                    color_dict[tuple(color)]=[(x,y)]
                    bbox_dict[tuple((x,y))]=[(xLeftBottom,yLeftBottom),(xRightTop,yRightTop)]

                else:
                    centroid = np.array([[[x,y]]],dtype=np.float32)
                    centroids = np.append(centroids,centroid,axis = 0)
                    color_dict[tuple(color)]=[(x,y)]
                    bbox_dict[tuple((x,y))]=[(xLeftBottom,yLeftBottom),(xRightTop,yRightTop)]

        original_centroids = centroids ########
        # else:
        #     color_dict=dict()

    else:   # track an object only if it has been detected
        if centroids.sum() != 0 and tracking_started:
            next1, st, error = cv2.calcOpticalFlowPyrLK(prev_frame, frame,
                                            centroids, None, **lk_params)

            good_new = next1[st==1]
            good_old = centroids[st==1]


            # print("color dict", color_dict)
            old_centroids = centroids

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                # Returns a contiguous flattened array as (x, y) coordinates for new point
                a, b = new.ravel()
                c, d = old.ravel()
                distance = np.sqrt((a-c)**2 + (b-d)**2)
                # distance between new and old points should be less than
                # 200 for 2 points to be same the object
                if distance < 200 :
                    corn_dict['centroids'][corn_dict[i]['centroid'][0]].append((a,b))
                    for color, centroids_list in color_dict.items():
                        # print("centroid list", centroids_list)
                        for centroids in centroids_list:
                            if centroids==(c,d):
                                color_dict[color].append((a,b))
                                color_old = color
                                frame = cv2.circle(frame, (a, b), 15, color_old, -1)
                    for centroids, bbox in bbox_dict.items():
                        if centroids==(c,d):
                            bbox_coor = bbox
                    
                    #### how to contorl id?
                    res = tuple(map(operator.sub, (c,d),corn_dict[i]['centroid'][0]))
                    new_bbox_coor1 = tuple(map(operator.add, corn_dict[i]['bbox'][0], res))
                    new_bbox_coor2 = tuple(map(operator.add, corn_dict[i]['bbox'][1], res))
                    new_bbox_coor1 = tuple(map(int, new_bbox_coor1))
                    new_bbox_coor2 = tuple(map(int, new_bbox_coor2))

                    frame = cv2.rectangle(frame, new_bbox_coor1, new_bbox_coor2, color_old, thickness=2) ### added today                    
                    # frame = cv2.rectangle(frame, bbox_coor[0], bbox_coor[1], color_old, thickness=2) ### added today
                    # frame = cv2.circle(frame, (a, b), 15, color_old, -1)
                    frame = cv2.putText(frame, str(total_frames), (100,100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 10, cv2.LINE_AA)

            centroids = good_new.reshape(-1, 1, 2)


        # print(corn_dict)
        # break

    total_frames += 1
    fps.update()
    fps.stop()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    if args.output:
        writer.write(frame)
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    prev_frame = frame
    if cv2.waitKey(1) >= 0:  # Break with ESC 
        break
