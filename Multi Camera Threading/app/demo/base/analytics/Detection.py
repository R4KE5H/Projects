import cv2
import numpy as np


class Detection:

    def __init__(self, cfg, weight, classes):
        self.network = cv2.dnn.readNetFromDarknet(cfg,weight)
        self.classes = classes
        self.layer_names = self.network.getLayerNames()
        self.outputlayers = [self.layer_names[i - 1] for i in self.network.getUnconnectedOutLayers()]

    def Detectionboxes(self,box):
        x, y, w, h = box
        return int(x), int(y), int(x + w), int(y + h)

    def detect(self, img, conf=0.2, nms_thresh=0.3, NMS=True, class_conf=None):
        if class_conf is None:
            class_conf = []
        if len(class_conf) < len(self.classes):
            conf = [conf] * len(self.classes)
        else:
            conf = class_conf
        class_conf = {k: conf[i] for i, k in enumerate(self.classes)}
        detection_output = {k: [] for k in self.classes}
        confidences = {k: [] for k in self.classes}
        boxes = {k: [] for k in self.classes}
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.network.setInput(blob)
        outs = self.network.forward(self.outputlayers)
        Height, Width, _ = img.shape
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > conf[int(class_id)]:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - (w / 2)
                    y = center_y - (h / 2)
                    confidences[self.classes[class_id]].append(float(confidence))
                    boxes[self.classes[class_id]].append([int(i) for i in [x, y, w, h]])
        indices = {}
        if NMS:
            for class_name, box in boxes.items():
                indices[class_name] = cv2.dnn.NMSBoxes(box, confidences[class_name], class_conf[class_name],
                                                       nms_thresh)
        else:
            for class_name, box in boxes.items():
                indices[class_name] = [[w] for w in range(len(box))]
        
        for key, index in indices.items():
            for i in index:
                select = i
                detection_output[key].append(self.Detectionboxes(boxes[key][select]))

        return detection_output
