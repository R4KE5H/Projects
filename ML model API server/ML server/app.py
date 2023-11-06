from flask import Flask, request, jsonify
from ultralyticsplus import YOLO
import cv2, base64, io, json
import numpy as np
from PIL import Image
from imageio import imread
from tracker import EuclideanDistTrackerAnalytics

app = Flask(__name__)


yoloV8 = YOLO('yolov8n')
yoloV8.overrides['conf'] = 0.25  # NMS confidence threshold
yoloV8.overrides['iou'] = 0.45  # NMS IoU threshold
yoloV8.overrides['agnostic_nms'] = False  # NMS class-agnostic
yoloV8.overrides['max_det'] = 1000  # maximum number of detections per image
yoloV8limit=0
limit_=50

camera_tracker={}

@app.route("/yolov8n",methods=['POST'])
def yolov8():
    global yoloV8, yoloV8limit, camera_tracker

    data = request.data
    # print(eval(str(data, 'UTF-8')).values())
    if list(eval(str(data, 'UTF-8')).keys())[0] not in camera_tracker:camera_tracker.update({list(eval(str(data, 'UTF-8')).keys())[0]:EuclideanDistTrackerAnalytics()})
    decode_img=base64.b64decode(list(eval(str(data, 'UTF-8')).values())[0])
    image = imread(io.BytesIO(decode_img))

    if yoloV8limit<limit_:
        results = yoloV8.predict(image, imgsz=640, classes = [2])
        # parse results
        result = results[0]
        boxes = result.boxes.xyxy # x1, y1, x2, y2
        scores = result.boxes.conf
        categories = result.boxes.cls
        scores = result.probs # for classification models
        masks = result.masks # for segmentation models
        limitcall=limit_-yoloV8limit
        yoloV8limit+=1
        #tracking
        boxes_ids = camera_tracker[list(eval(str(data, 'UTF-8')).keys())[0]].update(result.boxes.xyxy.tolist())


        return "{}:{}".format(limitcall,["yoloV8",result.boxes.xyxy,boxes_ids])
        # return "<h1>{} Api call Limit: {}</h1>".format(result.boxes.xyxy, limitcall)
    else:
        return "None"
        # return "API Limit Exceeded" 


@app.route('/test', methods=['POST'])
def test():
    data = request.data
    decode_img=base64.b64decode(list(eval(str(data, 'UTF-8')).values())[0])
    image = imread(io.BytesIO(decode_img))
    cv2.imshow("frame",image)
    cv2.waitKey(10)
    return "None"



