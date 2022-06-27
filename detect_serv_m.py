# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import time
from pathlib import Path

import uvicorn
import argparse
import os
import sys
from pathlib import Path

import matplotlib.path as mpltPath
from matplotlib.patches import PathPatch




import json
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.datasets import letterbox
from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import apply_classifier, check_img_size, check_imshow, check_requirements, check_suffix, colorstr, \
    increment_path, non_max_suppression, print_args, save_one_box, scale_coords, set_logging, \
    strip_optimizer, xyxy2xywh
from utils.plots import Annotator, colors
from utils.torch_utils import load_classifier, select_device, time_sync

from utils.datasets2 import *
from utils.utils2 import *

from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError, validator
from typing import List, Optional

from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

import torch
import base64
import random

app = FastAPI()










update = False

imgsz = 640
save_txt = True
view_img = False
weights = 'deployment_model/deployment.pt'
save_img = True #not opt.nosave and not source.endswith('.txt')  # save inference images
exist_ok = True

conf_thres = 0.25
iou_thres = 0.45
classes=None
agnostic_nms = False
max_det = 1000
save_conf = False 
hide_conf = False
hide_labels = False
line_thickness = 3
save_crop = False


# Directories
save_dir = increment_path(Path("runs/detect/exp"), exist_ok=exist_ok)  # increment run
(save_dir / 'labels' ).mkdir(parents=True, exist_ok=True)  # make dir

half2 = False
dev = os.environ['DEVICE']  #"cpu"
set_logging()
device = select_device(dev)
half = half2 and device.type != 'cpu'  # half precision only supported on CUDA
   
# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size
names = model.module.names if hasattr(model, 'module') else model.names  # get class names

if half:
    model.half()  # to FP16

if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once


#global distance_w,distance_h
#points = [(564, 944), (1642, 944), (1480, 327), (1141, 327), (1084, 916), (1379, 924), (1124, 793), (1367, 769)]
#distance_w, distance_h = (542.7538529396854,84.75061615739091)

# Define the Input class
class Image(BaseModel):
    base64: str
    name: str
    imgID: str
    checkfor:str
    ROI:list
    enSave:int

# Define the InputSD class for detecting social distancing
class ImageSD(BaseModel):
    base64: str
    points: list 
    name: str
    imgID: str
    checkfor:str
    ROI:list
    enSave:int

# we are encoding image file to a Base64 String, then decode it to retrieve the original image
# we run the model on the image file to get the predictions.
# We use Non max suppression technique for selecting the best bounding box out of a set of overlapping boxes.

@app.post("/detect")
async def index(request: Image):
    """
    This function will take a base 64 image as input and return a json object

    Args:
        request : Image

    Returns:
        json object

    """
    img_b = request.base64
    cam_name = request.name
    imgID = request.imgID 
    checkfor = request.checkfor
    checkfor = checkfor.split(":")
    roi = request.ROI
    enSave = request.enSave
    roi_points = []
    inTime = time.time()
    print("Camera ID:: ",cam_name)
    print("in time:: ", inTime)
    
    if roi[0] != "NA":
        for num in range(len(roi)):
            roi_points.append(tuple(map(int, roi[num].split(','))))
            print(roi_points[num])
        print("roi::",roi_points)
    
    path = mpltPath.Path(roi_points)

    
    img_data = base64.b64decode(img_b)
    jpg_as_np = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    img0 = img    
    img = letterbox(img, 640, 32)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    augment = False
    t1 = time_sync()
    pred = model(img, augment=augment)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    print(pred)
    lab = ''
    outdata = {}
    annotator = Annotator(img0, line_width=line_thickness, example=str(names))
    
    pts = np.array(roi_points)
    isClosed = True
    color = (34,65,200)
    thickness = 5
 
    #polylines(img, pts, isClosed, color[, thickness[, lineType[, shift]]]) -> img
    cv2.polylines(img0, [pts], isClosed, color, thickness)
 
    #patch=PathPatch(path,alpha=0.5)
    #img0.add_patch(patch)
    for i, det in enumerate(pred):  # detections per image
        count = 0
        #outdata["totalDetection"]=str(len(det))
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        rect=[]
        if len(det):
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            s=''
            validDetections = []
            # Print results
#            for c in det[:, -1].unique():
#                n = (det[:, -1] == c).sum()  # detections per class
#                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
#            outdata["DetectionPerClass"]=s
            #count = 0
            # Write results
            for *xyxy, conf, cls in reversed(det):
                inside =  False
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}')

                width, height, col = img0.shape
                xyxy_t = [int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])]
                
                if path.contains_points([[xyxy_t[0],xyxy_t[1]]]) or path.contains_points([[xyxy_t[2],xyxy_t[3]]]):
                    inside = True
                
                if inside:
                    count = count +1
                    outdata["Detection id "+str(count)+" "+str(label)]=str(int(xyxy_t[0]))+","+str(int(xyxy_t[1]))+","+str(int(xyxy_t[2]))+","+str(int(xyxy_t[3]))
                    detected = label.lower()
                    detected_class = list(label.lower())
                    if detected in checkfor:
                        if detected_class[0] =='n' and detected_class[1]=='o':
                            annotator.box_label(xyxy, label, color=(0,0,255))
                        else:
                            annotator.box_label(xyxy, label, color=(0,255,0))
                             
                        validDetections.append(label)

            for c in set(validDetections[:]):
                n = validDetections[:].count(c)  # detections per class
                s += f"{n} {c}{'s' * (n > 1)}, "  # add to string
            outdata["DetectionPerClass"]=s

        im0 = annotator.result()
        outdata["totalDetection"]=str(count)
        if enSave:
            fileName = 'runs/output_'+str(int(inTime))+'_'+cam_name+'_'+imgID+'.jpg'
            cv2.imwrite(fileName,img0)
        retval, buffer = cv2.imencode('.jpg', img0)
        b64string = base64.b64encode(buffer).decode("utf-8")
        outdata["outimage"]=b64string
        #cv2.imwrite('runs/output_'+cam_name+'_'+imgID+'.jpg', img0)

    outTime = time.time()
    print("out time:: ", outTime)
    print("Processing Time::",outTime-inTime)

    json_outdata = json.dumps(outdata)
    return json_outdata

@app.post("/socialdist")
async def index(request: ImageSD):
    """
    This function will take a base 64 image as input and return a json object

    Args:
        request : ImageSD

    Returns:
        json object

    """
    img_b = request.base64
    t = request.points
    cam_name = request.name
    imgID = request.imgID
    checkfor = request.checkfor
    checkfor = checkfor.split(":")
    roi = request.ROI
    enSave = request.enSave
    points = []
    roi_points = []
    inTime = time.time()
    print("Camera ID:: ",cam_name)
    print("in time:: ", inTime)
    for num in range(len(t)):
        points.append(tuple(map(int, t[num].split(','))))
        print(points[num])
    print("points::",points)
    
    if roi[0] != "NA":
        for num in range(len(roi)):
            roi_points.append(tuple(map(int, roi[num].split(','))))
            print(roi_points[num])
        print("roi::",roi_points)
    
    path = mpltPath.Path(roi_points)
    img_data = base64.b64decode(img_b)
    jpg_as_np = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    img0 = img    
    img = letterbox(img, 640, 32)[0]
    
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    # Inference
    augment = False
    t1 = time_sync()
    pred = model(img, augment=augment)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    print(pred)
    lab = ''
    outdata = {}
    annotator = Annotator(img0, line_width=line_thickness, example=str(names))
    # List to store bounding coordinates of people
    people_coords = []
    people_coords_bot = []
    SDcount = 0
    pts = np.array(roi_points)
    isClosed = True
    color = (34,65,200)
    thickness = 5
 
    #polylines(img, pts, isClosed, color[, thickness[, lineType[, shift]]]) -> img
    cv2.polylines(img0, [pts], isClosed, color, thickness)
    #patch=PathPatch(path,alpha=0.5)
    #img0.add_patch(patch)
    for i, det in enumerate(pred):  # detections per image
        count = 0
        #outdata["totalDetection"]=str(len(det))
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        rect=[]
        if len(det):
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            s=''
            validDetections = []
            # Print results
            #for c in det[:, -1].unique():
            #    n = (det[:, -1] == c).sum()  # detections per class
            #    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
            #outdata["DetectionPerClass"]=s
            # Write results
            for *xyxy, conf, cls in reversed(det):
                inside =  False
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format

                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}')
                width, height, col = img0.shape
                W = width
                H = height
                xyxy_t = [int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])]
                
                if path.contains_points([[xyxy_t[0],xyxy_t[1]]]) or path.contains_points([[xyxy_t[2],xyxy_t[3]]]):
                    inside = True
                
                
                if inside:
                    count = count +1
                    outdata["Detection id "+str(count)+" "+str(label)]=str(int(xyxy_t[0]))+","+str(int(xyxy_t[1]))+","+str(int(xyxy_t[2]))+","+str(int(xyxy_t[3]))
                    detected = label.lower()
                    detected_class = list(label.lower())
                    if detected in checkfor:
                        if detected_class[0] =='n' and detected_class[1]=='o':
                            annotator.box_label(xyxy, label, color=(0,0,255))
                        else:
                            annotator.box_label(xyxy, label, color=(0,255,0))
                    
                    if label is not None:
                        print(label)
                        if 'human' in (label.split())[0].lower():
                            print("human label is present")
                            people_coords.append(xyxy)
                            xywh_bot = (xyxy2xywh_bot(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                            print("x-x-x")
                            print(xywh_bot)
                            # plot_one_box(xyxy, im0, line_thickness=3)
                            people_coords_bot.append(xywh_bot)  
                            plot_dots_on_people(xyxy, img0)
                            annotator.box_label(xyxy, label, color=(0,255,0))
            

                            
                        validDetections.append(label)

            for c in set(validDetections[:]):
                n = validDetections[:].count(c)  # detections per class
                s += f"{n} {c}{'s' * (n > 1)}, "  # add to string
            outdata["DetectionPerClass"]=s

            # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are
            # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view.
            # This bird eye view then has the property that points are distributed uniformally horizontally and
            # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are
            # equally distributed, which was not case for normal view.
            src = np.float32(np.array(points[:4]))
            dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
            # Here we will be using bottom center point of bounding box for all boxes and will transform all those
            # bottom center points to bird eye view
            prespective_transform = cv2.getPerspectiveTransform(src, dst)

            if np.array(people_coords_bot).ndim > 1:
                print(people_coords_bot)
                pts = np.float32([np.array(people_coords_bot)[:,:2]])
                print(pts)
                person_points = cv2.perspectiveTransform(pts, prespective_transform)[0]

                print("Person Points")
                print(person_points)
                # Here we will calculate distance between transformed points(humans)
                # Plot Lines connecting people
                SDcount = distancing_bot(people_coords, person_points, img0, dist_thres_lim=(500,650))
            outdata["SDcount"]=SDcount
        outdata["totalDetection"]=str(count)
        retval, buffer = cv2.imencode('.jpg', img0)
        b64string = base64.b64encode(buffer).decode("utf-8")
        outdata["outimage"]=b64string
        im0 = annotator.result()
        if enSave:
            fileName = 'runs/output_'+str(int(inTime))+'_'+cam_name+'_'+imgID+'.jpg'
            cv2.imwrite(fileName,img0)
        #cv2.imwrite('runs/output_'+cam_name+'_'+imgID+'.jpg', img0)
    
    json_outdata = json.dumps(outdata)
    outTime = time.time()
    print("out time:: ", outTime)
    print("Processing Time::",outTime-inTime)
    return json_outdata

if __name__ == "__main__":
    
#    update = False
#    
#    imgsz = 640
#    save_txt = True
#    view_img = False
#    weights = 'runs/deployment.pt'
#    save_img = True #not opt.nosave and not source.endswith('.txt')  # save inference images
#    exist_ok = True
#    
#    conf_thres = 0.25
#    iou_thres = 0.45
#    classes=None
#    agnostic_nms = False
#    max_det = 1000
#    save_conf = False 
#    hide_conf = False
#    hide_labels = False
#    line_thickness = 3
#    save_crop = False
#    
#    
#    # Directories
#    save_dir = increment_path(Path("runs/detect/exp"), exist_ok=exist_ok)  # increment run
#    (save_dir / 'labels' ).mkdir(parents=True, exist_ok=True)  # make dir
#    
#    half2 = False
#    dev = "0"
#    set_logging()
#    device = select_device(dev)
#    half = half2 and device.type != 'cpu'  # half precision only supported on CUDA
#       
#    # Load model
#    model = attempt_load(weights, map_location=device)  # load FP32 model
#    stride = int(model.stride.max())  # model stride
#    imgsz = check_img_size(imgsz, s=stride)  # check img_size
#    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
#    
#    if half:
#        model.half()  # to FP16
#    
#    if device.type != 'cpu':
#        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
#    
#    #make the app string equal to whatever the name of this file is
    uvicorn.run(app, host='0.0.0.0', port=10230)
