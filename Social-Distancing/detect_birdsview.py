import argparse

from utils.datasets import *
from utils.utils import *
import cv2
import numpy as np
import time
import argparse

# own modules
import utills

confid = 0.5
thresh = 0.5
mouse_pts = []


# Function to get points for Region of Interest(ROI) and distance scale. It will take 8 points on first frame using mouse click    
# event.First four points will define ROI where we want to moniter social distancing. Also these points should form parallel  
# lines in real world if seen from above(birds eye view). Next 3 points will define 6 feet(unit length) distance in     
# horizontal and vertical direction and those should form parallel lines with ROI. Unit length we can take based on choice.
# Points should pe in pre-defined order - bottom-left, bottom-right, top-right, top-left, point 5 and 6 should form     
# horizontal line and point 5 and 7 should form verticle line. Horizontal and vertical scale will be different. 

# Function will be called on mouse events                                                          

def get_mouse_points(event, x, y, flags, param):

    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:
            cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), 10)
            
        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
            if len(mouse_pts) == 3:
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)
        
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))
        #print("Point detected")
        #print(mouse_pts)
        

def calibrate (img_path):
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    print(height,width,channels)

    H = height
    W = width
    scale_w, scale_h = utills.get_scale(width, height)

    points = []
    global image
    
    while True:
        image = img
        cv2.imshow("image", image)
        cv2.waitKey(1)
        if len(mouse_pts) == 8:
            cv2.destroyWindow("image")
            break
       
    points = mouse_pts
    print(points)
    # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are 
    # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view. 
    # This bird eye view then has the property property that points are distributed uniformally horizontally and 
    # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are 
    # equally distributed, which was not case for normal view.
    src = np.float32(np.array(points[:4]))
    dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
    prespective_transform = cv2.getPerspectiveTransform(src, dst)

    # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
    pts = np.float32(np.array([points[4:7]]))
    warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]
    
    # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
    # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
    # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
    # which we can use to calculate distance between two humans in transformed view or bird eye view
    global distance_w,distance_h
    distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
    distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
    pnts = np.array(points[:4], np.int32)
    print(distance_w,distance_h)
    cv2.polylines(image, [pnts], True, (70, 70, 70), thickness=2)


def detect(calib,save_img=False):
    
    global distance_w,distance_h
    points = [(564, 944), (1642, 944), (1480, 327), (1141, 327), (1084, 916), (1379, 924), (1124, 793), (1367, 769)]
    distance_w, distance_h = (542.7538529396854,84.75061615739091)
    img_path = "/home/vikram/Documents/TRAKR_AI/Social-Distancing/inference/inp.png"
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    print(height,width,channels)

    H = height
    W = width
    scale_w, scale_h = utills.get_scale(width, height)

    if calib:
        img_path = "/home/vikram/Documents/TRAKR_AI/Social-Distancing/inference/inp.png"
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        print(height,width,channels)

        H = height
        W = width
        scale_w, scale_h = utills.get_scale(width, height)

        points = []
        global image
        
        while True:
            image = img
            cv2.imshow("image", image)
            cv2.waitKey(1)
            if len(mouse_pts) == 8:
                cv2.destroyWindow("image")
                break
           
        points = mouse_pts
        print(points)
        # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are 
        # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view. 
        # This bird eye view then has the property property that points are distributed uniformally horizontally and 
        # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are 
        # equally distributed, which was not case for normal view.
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        prespective_transform = cv2.getPerspectiveTransform(src, dst)

        # using next 3 points for horizontal and vertical unit length(in this case 180 cm)
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]
        

        test_pts = np.float32(np.array([[(443.5,261),(297,296),(388,281),(455,112)]]))
        test_warped_pt = cv2.perspectiveTransform(test_pts, prespective_transform)[0]

        print("Warped Points")
        print(test_warped_pt)

        
        # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
        # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
        # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
        # which we can use to calculate distance between two humans in transformed view or bird eye view
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)

        print(distance_w,distance_h)
        cv2.polylines(image, [pnts], True, (70, 70, 70), thickness=2)
        


    
    
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   fast=True, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # List to store bounding coordinates of people
        people_coords = []
        people_coords_bot = []

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        if label is not None:
                            if (label.split())[0] == 'person':
                                people_coords.append(xyxy)
                                
                                xywh_bot = (xyxy2xywh_bot(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # normalized xywh
                                #xywh_bot = (xyxy2xywh_bot(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                #xywh_bot = (xyxy2xywh_bot(torch.tensor(xyxy).view(1, 4)))
                                print("x-x-x")
                                print(xywh_bot)
                                # plot_one_box(xyxy, im0, line_thickness=3)
                                people_coords_bot.append(xywh_bot)  
                                plot_dots_on_people(xyxy, im0)
            #print("People coordinates")
            #print(people_coords_bot)
            
            #box = detection[0:4] * np.array([W, H, W, H])
            #(centerX, centerY, width, height) = box.astype("int")

            #x = int(centerX - (width / 2))
            #y = int(centerY - (height / 2))

            #boxes.append([x, y, int(width), int(height)])
            
            
            # Here we will be using bottom center point of bounding box for all boxes and will transform all those
            # bottom center points to bird eye view
            #person_points = utills.get_transformed_points_bot(people_coords_bot, prespective_transform)
            #prespective_transform = cv2.getPerspectiveTransform(src, dst)
            #if len(det)>1:
                

            # Calibration points
                #points = [(568, 957), (1652, 958), (1482, 339), (1171, 333), (1067, 918), (1337, 907), (1116, 791), (1347, 784)]

                src = np.float32(np.array(points[:4]))
                dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
                prespective_transform = cv2.getPerspectiveTransform(src, dst)

                if np.array(people_coords_bot).ndim > 1:
                    print(people_coords_bot)
                    pts = np.float32([np.array(people_coords_bot)[:,:2]])
                    print(pts)
                    person_points = cv2.perspectiveTransform(pts, prespective_transform)[0]

                    print("Person Points")
                    print(person_points)
                #exit()
                #print(person_points)
                
                # Here we will calculate distance between transformed points(humans)
                #distances_mat, bxs_mat = utills.get_distances(boxes1, person_points, distance_w, distance_h)
                #risk_count = utills.get_count(distances_mat)
                
                #print(risk_count)
                
                # Plot lines connecting people
                    distancing_bot(people_coords, person_points, im0, dist_thres_lim=(500,650))
                #distancing(people_coords, people_coords_bot, prespective_transform, im0, dist_thres_lim=(100,150))

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/yolov5s.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('-cal', '--calibrate', action='store', dest='calib', default=False ,
                    help='Path for input video')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", get_mouse_points)
    np.random.seed(42)
    calib = opt.calib
    if calib:
        print(calib)
        calibrate("/home/vikram/Documents/TRAKR_AI/Social-Distancing/inference/inp.png")
    with torch.no_grad():
        detect(calib)
