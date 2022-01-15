import numpy as np
from yolov3_pytorch import *







weightsfile = '.\\yolov3\yolov3.weights'
classfile = '.\coco.names'
cfgfile = '.\\yolov3\yolov3.cfg'
# sample_img1 = '.\data\yolo\input\dog-cycle-car.png'
input_dir="C:\\Users\\RAKESH\\Pictures\\batch_inferencing\\"
nms_thesh = 0.5




CUDA = False
batch_size = 11
#Set up the neural network
print("Loading network.....")
model = Darknet(cfgfile)
model.load_weights(weightsfile)
print("Network successfully loaded")
classes = load_classes(classfile)
print('Classes loaded')
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32

#If there's a GPU availible, put the model on GPU
if CUDA:
    model.cuda()

#Set the model in evaluation mode
model.eval()


# read images from folder 'images' or direcly  image
read_dir = time.time()
#Detection phase
try:
    imlist = [os.path.join(os.path.realpath('.'), input_dir, img) for img in os.listdir(input_dir)]
except NotADirectoryError:
    imlist = []
    imlist.append(os.path.join(os.path.realpath('.'), input_dir))
except FileNotFoundError:
    print ("No file or directory with the name {}".format(input_dir))
    exit()
    

    
load_batch = time.time()

# preparing list of loaded images
# [[image,original_image,dim[0],dim[1]]]
batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
im_batches = [x[0] for x in batches] # list of resized images
orig_ims = [x[1] for x in batches] # list of original images
im_dim_list = [x[2] for x in batches] # dimension list
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2) #repeating twice
    
    
if CUDA:
    im_dim_list = im_dim_list.cuda()

# converting image to batches    
reminder = 0
if (len(im_dim_list) % batch_size): #if reminder is there, reminder = 1
    reminder = 1

# if batch_size != 1:
#     num_batches = len(imlist) // batch_size + reminder            
#     im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,len(im_batches))])) 
#                  for i in range(num_batches)] 

    
i = 0
write = False
    
objs = {}    
count= 1
print("Image batch: ", len(im_batches))
start_time = time.time()
print("Detection time starts..")
for batch in im_batches:
        #load the image 
        start = time.time()
        if CUDA:
            batch = batch.cuda()       
        #Apply offsets to the result predictions
        #Tranform the predictions as described in the YOLO paper
        #flatten the prediction vector 
        # B x (bbox cord x no. of anchors) x grid_w x grid_h --> B x bbox x (all the boxes) 
        # Put every proposed box as a row.
        with torch.no_grad():
            prediction = model(Variable(batch,volatile = True), CUDA)
            count+=1
        prediction, dummy = write_results(prediction, confidence=0.5, num_classes=80, nms_conf = nms_thesh)
        if type(prediction) == int or dummy == 0:
            i += 1
            continue

        prediction[:,0] += i #*batch_size
        if not write:
            output = prediction
            write = 1
        else:
            output = torch.cat((output,prediction))  # concating predictions from each batch
        i += 1
        
        if CUDA:
            torch.cuda.synchronize()
    
try:
    output
except NameError:
    print("No detections were made")
    exit()

    
#Before we draw the bounding boxes, the predictions contained in our output tensor 
#are predictions on the padded image, and not the original image. Merely, re-scaling them 
#to the dimensions of the input image won't work here. We first need to transform the
#co-ordinates of the boxes to be measured with respect to boundaries of the area on the
#padded image that contains the original image

im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
output[:,1:5] /= scaling_factor
    
for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
        
            

def write(x, batches, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])]
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = (0,0,255)
    cv2.rectangle(img, (int(c1[0]),int(c1[1])), (int(c2[0]),int(c2[1])),color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, (int(c1[0]),int(c1[1])), (int(c2[0]),int(c2[1])),color, -1)
    cv2.putText(img, label, (int(c1[0]), int(c1[1]) + int(t_size[1]) + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img


for x in output:
    ret_img=write(x, im_batches, orig_ims)
    cv2.imshow("sample",ret_img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
print("total time taken: ", time.time()-start_time)    

    


cv2.destroyAllWindows() 
torch.cuda.empty_cache()