import time
import os, cv2
import tqdm
import numpy as np
import torch
from torch.autograd import Variable
from yolo_detectorPytorch import load_model
from utils import load_classes, non_max_suppression, rescale_boxes

img_size=416 #change the value according the input shape
cfg_path=r'C:\Users\RAKESH\Desktop\Personal\Study\pytorch\data\yolo\yolov3_tiny\yolov3-tiny.cfg'
weights_path=r'C:\Users\RAKESH\Desktop\Personal\Study\pytorch\data\yolo\yolov3_tiny\yolov3-tiny.weights'
input_path=r"C:\\Users\\RAKESH\\Pictures\\batch_inferencing\\"
classes_path=r'C:\Users\RAKESH\Desktop\Personal\Study\pytorch\data\yolo\coco.names'
conf_thres=0.5
nms_thres=0.4
cv2_display=True
batch_size=11
results={}

def draw_on_frame(img, result):
    for cls, objs in result.items():
        for x1, y1, x2, y2 in objs:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img,cls , (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=3)  
    return img
        
    
def detect(model, images, conf_thres, nms_thres):
    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_detections = []  # Stores detections for each image index
    imgs_index = []  # Stores image paths
    start_time=time.time()
    count=0
    for input_imgs in tqdm.tqdm(images, desc="Detecting"):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))
        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Store image and detections
        img_detections.extend(detections)
        imgs_index.append(count)
        count+=1
    print("Time taken: ", time.time()-start_time,"Length of batch: ",len(img_detections))    
    return img_detections, imgs_index

def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,:] = resized_image
    
    return canvas

def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = cv2.imread(img)
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def dataset(input_dir):
    global img_size, batch_size
    try:
        img_list = os.listdir(input_dir)
        imlist = [os.path.join(os.path.realpath('.'), input_dir, img_list[img]) for img in range(batch_size)]
    except NotADirectoryError:
        imlist = []
        imlist.append(os.path.join(os.path.realpath('.'), input_dir))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(input_dir))
        exit()
    
    inp_dim = img_size
    assert inp_dim % 32 == 0 
    assert inp_dim > 32
    
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches] # list of resized images
    orig_ims = [x[1] for x in batches] # list of original images
    # im_dim_list = [x[2] for x in batches] # dimension list
    # im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2) #repeating twice    
    
    return im_batches, orig_ims  

def detect_directory(model_path, weights_path, img_path, classes,
                      conf_thres=0.5, nms_thres=0.5):
    global cv2_display, results, img_size
    images, ori_img = dataset(input_dir=img_path)
    model = load_model(model_path, weights_path)
    img_detections, imgs_index = detect(model,images,conf_thres,nms_thres)
    if cv2_display:
        for boxes,j in zip(img_detections, imgs_index):
            for cls in classes:
                results[cls]=[]    
            boxes=rescale_boxes(boxes, img_size, ori_img[j].shape[:2])
            boxes=boxes.int().tolist()
            if len(boxes)!=0:
                for box in boxes:
                    results[classes[box[-1]]].append(box[0:4])
            frame=draw_on_frame(ori_img[j], results) 
            frame=cv2.resize(frame,(680,420))
            cv2.imshow("test", frame)       
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
                    
def run():
    global cfg_path, weights_path, classes_path, conf_thres, nms_thres, input_path
    classes = load_classes(classes_path)  # List of class names
    detect_directory(cfg_path,weights_path,input_path,classes,conf_thres=conf_thres,nms_thres=nms_thres)

if __name__ == '__main__':
    run()
