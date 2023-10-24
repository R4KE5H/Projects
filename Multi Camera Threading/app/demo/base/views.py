from django.shortcuts import render
from django.http import HttpResponse
from django.shortcuts import redirect, render
import json, cv2
from .analytics.Detection import Detection
from .analytics.tracker import EuclideanDistTracker
import threading
from django.http.response import StreamingHttpResponse
import numpy as np
import os, time

print(threading.active_count())

class Streaming():
    """
    This class function is used to store the each camera details
    """    
    streams={}
    def __init__(self, path:str, weights:str, cfg:str, classes_path:str,camera_name:str):
        self.camera_name=camera_name
        with open(classes_path) as f:
            lines = f.readlines()
        f.close()    
        self.classes=[cls.strip("\n") for cls in lines]
        self.detector = Detection(cfg, weights, self.classes)
        self.cap = cv2.VideoCapture(path)
        self.tracker=EuclideanDistTracker()
        self.frame=np.array([])
        self.trackers = []
        self.labels=[]


        Streaming.streams.update({camera_name:self})
    
    @staticmethod
    def draw_on_frame(frame,boxes_ids):
        """
        Args:
            frame ([numpy array]): Receives the image from video source
            boxes_ids ([list]): Receives the detcetion co ordinate 

        Returns:
            [numpy array]: Returns the frame with detected bounding boxes
        """        
        for box_id in boxes_ids:
            x1,y1,x2,y2,id = box_id
            cv2.putText(frame, str(id),(x1,y1-15),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.rectangle(frame, (x1,y1),(x2, y2), (0,255,0), 2)

        return frame    


    def read_frame(self):
        """
        Read the frame and detect the trained objects with tracking id using euclidean distance 
        """        
        while self.cap.isOpened():
            start_time=time.time()
            ret, frame = self.cap.read()
            if ret==False:
                self.frame=cv2.imread("./dummy.jpg")
            h, w, _ = frame.shape
            results = self.detector.detect(frame, conf=0.3)
            boxes_ids = self.tracker.update(results["car"])  
            frame = self.draw_on_frame(frame,boxes_ids)
            try:cv2.putText(frame, "fps: {}".format(1 / (time.time() - start_time)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0),thickness=1)
            except:pass    
            self.frame=frame                   
            # cv2.imshow(self.camera_name, cv2.resize(self.frame, (640,420)))
            # cv2.waitKey(10)
        


       
def main():
    """
    This function is used to initialize the camera and its thread
    """    
    f = open("./base/settings/setting.json")
    data = json.load(f)
    f.close()
    for cam, attr in data.items():
        if attr["stream"]:
            Streaming(attr["stream"],attr["yolo"]["weights"],attr["yolo"]["cfg"],attr["yolo"]["classes"],cam)

    thread_ls=[]
    for cam_n, run_cam in Streaming.streams.items():
        thread_ls.append(threading.Thread(target=run_cam.read_frame, args=()))

    for th in thread_ls:
        th.daemon = True
        th.start()


if True:
    """
    Initializing the main function
    """    
    main()


def index(request):
    """
    Args:
        request ([HttpRequest]): Recieves HttpRequest

    Returns:
        [HttpResponse]: Return HttpResponse 
    """    
    return render(request, 'index/index1.html')


def index2(request):
    """
    Args:
        request ([HttpRequest]): Recieves HttpRequest

    Returns:
        [HttpResponse]: Return HttpResponse 
    """ 
    return render(request, 'index/index2.html')


def send_images(id_ref):
    """
    Args:
        id_ref ([str]): Receives the camera id 

    Yields:
        [str]: Returns encoded image
    """    
    while True:
        ret, buffer = cv2.imencode(".jpg", Streaming.streams[id_ref].frame if Streaming.streams[id_ref].frame.any() else cv2.imread("./dummy.jpg") )
        img = buffer.tobytes()
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + img + b'\r\n'

def image_stream(request, id_ref):
    """
    Args:
        request ([HttpRequest]): Recieves HttpRequest
        id_ref ([str]): Recieves reference id

    Returns:
        [type]: [description]
    """    
    return StreamingHttpResponse(send_images(id_ref),
                                 content_type='multipart/x-mixed-replace; boundary=frame')