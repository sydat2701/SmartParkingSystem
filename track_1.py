# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams, VID_FORMATS
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

from threshSps import list_x, list_y, list_poly, list_poly_parking, list_gates, list1, list2, list3
from shapely.geometry import Point, Polygon
from FindPath_Multiroad import FindPath, FillGrid
from FindPath_Multiroad.utils import *
from FindPath_Multiroad.FindPath import *
from hyperParams import *
import FindPath_Multiroad.CheckPointInPolygonCplusplus as PIPolygon
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

list_cars = [] #danh sach de luu cac xe cho viec kiem tra vao cong
dict_coor_of_car={}     #su dung cho viec check ra vao cong
list_cars_checkin=[] #danh sach cac xe thuc su vua vao cong
#---------------------------------------------------------------------------------
def checkCoordinates(x,y):
    x_1,y_1=x.cpu().numpy(), y.cpu().numpy()
    if len(list_x)>0:
        for ele in list_x:
            if x_1<ele[0] or x_1>ele[1]:
                return False
            return True
    #print("**********************", x_1,y_1)
    if len(list_y)>0:
        for ele in list_y:
            #print("^^^^^^^", ele[0], ele[1])
            if y_1<ele[0] or y_1>ele[1]:
                return False
            return True
    return True

# print("///////////////////////////////////////////////////////list2")
# print(list1_check)

#_---------------------------------------------------------------------------------

def checkWithin(x,y):
    x_1,y_1=x.cpu().numpy(), y.cpu().numpy()
    #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #print(list1_check[0])
    
    for poly in list1_check:
        #point =Point(x_1, y_1)
        #if point.within(poly):
        #print("##############################################################")
        #sprint(list1_check)
        #print(poly)
        if PIPolygon.PointInPolygon((x_1,y_1), poly):
            #print(":::::::::::::::::::::", x_1, y_1, poly)
            return True
    return False

#----------------------------------------------------------------------------
def checkWithinParking(x,y):
    try:
        x_1,y_1=x.cpu().numpy(), y.cpu().numpy()
    except:
        x_1, y_1=x,y
    for poly in list2_check:
        #point =Point(x_1, y_1)
        if PIPolygon.PointInPolygon((x_1, y_1), poly):
            return True
    return False

#-----------------------------------check xe vao cong------------------------------------------------

def checkInGatePoly1(x,y):
    x_1,y_1=x, y
    for polies in list3_check:
        poly1=polies[0]
        #point =Point(x_1, y_1)
        #if point.within(poly1):
        if PIPolygon.PointInPolygon((x_1,y_1), poly1):
            return True
    return False

def checkInGatePoly2(x,y):
    x_1,y_1=x,y
    for polies in list3_check:
        poly2=polies[1]
        #point =Point(x_1, y_1)
        #if point.within(poly2):
        if PIPolygon.PointInPolygon((x_1, y_1), poly2):
            return True
    return False



#--------------------------------------------------------------------------
list1_check=[]
list2_check=[]
list3_check=[]

if xy_inverse==True:
    for rd in list1:
        tmp_rd=[]
        for vertex in rd:
            tmp_rd.append([vertex[1], vertex[0]])
        list1_check.append(tmp_rd)

    for rd in list2:
        tmp_rd=[]
        for vertex in rd:
            tmp_rd.append([vertex[1], vertex[0]])
        list2_check.append(tmp_rd)
    
    list3_check=list3.copy()
    
    # for gt in list3:
    #     tmp_gt=[]
    #     for area in gt:
    #         tmp_area=[]
    #         for vertex in area:
    #             tmp_area.append([vertex[1], vertex[0]])
    #         tmp_gt.append(tmp_area)
    #     list3_check.append(tmp_gt)


# print("**********************************************")
# print(list1_check)
# print(list2_check)
# print("--------------")
# print(list3_check)
time_find_path=0

#--------------------------------------------------------------------------
import time
def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, \
        project, exist_ok, update, save_crop = \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.exist_ok, opt.update, opt.save_crop
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    if type(yolo_model) is str:  # single yolo model
        exp_name = yolo_model.split(".")[0]
    elif type(yolo_model) is list and len(yolo_model) == 1:  # single models after --yolo_model
        exp_name = yolo_model[0].split(".")[0]
    else:  # multiple models after --yolo_model
        exp_name = "ensemble"
    exp_name = exp_name + "_" + deep_sort_model.split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run if project name exists
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    #print("::::::::::::::::::::::::;;img size:", imgsz)

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)

    # Create as many trackers as there are video sources
    deepsort_list = []
    for i in range(nr_sources):
        deepsort_list.append(
            DeepSort(
                deep_sort_model,
                device,
                max_dist=cfg.DEEPSORT.MAX_DIST,
                max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            )
        )
    outputs = [None] * nr_sources

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", )
    # print("###########################################")
    # print(list1_check)
    #-------------------------------------------------------------------------------------------
    # if xy_inverse==True:
    #     for rd in list1:
    #         for vertex in rd:
    #             tg=vertex[0]
    #             vertex[0]=vertex[1]
    #             vertex[1]=tg
    #     for rd in list2:
    #         for vertex in rd:
    #             tg=vertex[0]
    #             vertex[0]=vertex[1]
    #             vertex[1]=tg

    # print("###########################################-----------------------")
    # print(list1_check)

        
    #--------------------------------------------------------------------------------------------

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0

    time1=time.time()
    time2=None
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        coor_car_to_track=[]#---------------
        coor_slots_to_track=[]#--------------

        t1 = time_sync()
        #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^", im.shape)
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
        pred = model(im, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        #print("+++++++++++++++++++++++++++++", opt.classes)    #[0,2]
        # pred_trong = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, 0, opt.agnostic_nms, max_det=opt.max_det)
        # pred_xe=non_max_suppression(pred, opt.conf_thres, opt.iou_thres, 2, opt.agnostic_nms, max_det=opt.max_det)
        pred= non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        tmp=pred[0].cpu().numpy()

        pred_trong=tmp[tmp[:,5]==0]
        pred_trong=torch.from_numpy(pred_trong)
        pred_trong=torch.unsqueeze(pred_trong, 0)

        pred_xe=tmp[tmp[:,5]==2]
        pred_xe=torch.from_numpy(pred_xe)
        pred_xe=torch.unsqueeze(pred_xe, 0)

        dt[2] += time_sync() - t3

        #-------------------------------------------------------------------------------------
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # print(pred_xe)
        # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        tmp_pred = []
        for ele in pred_xe[0]:
            #print("::::::::::::::::::::::::", ele)
            x, y, xx, yy= ele[0], ele[1], ele[2], ele[3]
            check=checkWithin(x, y)

            if check:
                tmp_pred.append(ele.cpu().numpy())

        # global pred_cars_in_polygon
        # pred_cars_in_polygon= tmp_pred

        #slots_to_track=[]

        #--------------------------- cho trong------------------------------------------
        for ele in pred_trong[0]:
            #print("::::::::::::::::::::::::", ele)
            x, y, xx, yy= ele[0], ele[1], ele[2], ele[3]
            #print("========================trong: ", x,y,width, height)
            check=checkWithinParking(x, y)

            if check:
                #print("*****************************************")
                tmp_pred.append(ele.cpu().numpy())
                x1=x.cpu().numpy()
                y1=y.cpu().numpy()
                xx1=xx.cpu().numpy()
                yy1=yy.cpu().numpy()
                #width1=width.cpu().numpy()
                #height1=height.cpu().numpy()
                coor_slots_to_track.append([ int(y1), int(x1), int(yy1), int(xx1)])

        #-------------------------------------------------------------------
        pred=torch.from_numpy(np.array(tmp_pred))
        pred=torch.unsqueeze(pred, 0)
                
        #--------------------------------------------------------------------------------------
        #car_to_track=[]

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to deepsort
                t4 = time_sync()
                outputs[i] = deepsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output) in enumerate(outputs[i]):

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]

                        #------------------------------------------------------------
                        x_yolo=output[0]+(output[2] - output[0])//2
                        y_yolo=output[1]+(output[3] - output[1])//2

                        # car_to_track.append([y_yolo, x_yolo])

                        # print("^^^^^^^^^^^^^^11111111^^^^^^^^^^^^^^^^^^^^^")
                        # print(x_yolo, y_yolo, id)
                        # print(checkInGatePoly1(x_yolo, y_yolo))
                        # print(list_cars)
                        # print(id not in list_cars)
                        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

                        if checkInGatePoly1(x_yolo, y_yolo) and id not in list_cars:
                            list_cars.append(id)
                            #print("****************************",list_cars)
                            #print("^^^^^^^^^^^^^^^^^^^Check1")

                        # print("^^^^^^^^^^^^^^^222222^^^^^^^^^^^^^^^^^^^^")
                        # print(x_yolo, y_yolo, id)
                        # print(checkInGatePoly2(x_yolo, y_yolo))
                        # print(list_cars)
                        # print(id in list_cars)
                        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

                        if checkInGatePoly2(x_yolo, y_yolo) and id in list_cars:
                            if id not in list_cars_checkin:
                                #print("//////////////////////////////////////",y_yolo, x_yolo)
                                list_cars_checkin.append(id)
                                #print("^^^^^^^^^^^^^^^^check2")
                                #print('++++++++++++++++++++++++++++here++++++++++++++++++++++++++++++++++++++++++++==')
                                #print(car_to_track)
                                #print(list_cars_checkin)
                            
                            list_cars.remove(id)
                        if id in list_cars_checkin:
                            coor_car_to_track.append([y_yolo, x_yolo])
                        
                        if (id in list_cars_checkin) and checkWithinParking(x_yolo, y_yolo):
                            coor_car_to_track.remove([y_yolo, x_yolo])
                            list_cars_checkin.remove(id)

                        #------------------------------------------------------------
                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, cls))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            # if (id not in list_cars_checkin) and (cls !=0):
                            #     continue
                            c = int(cls)  # integer class
                            label = f'{id:0.0f} {names[c]} {conf:.2f}'
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                        
                        # if len(coor_car_to_track)==0:
                        #     print(list_cars_checkin)
                        #     print(coor_car_to_track)
                        #     print("looooooooooooooooooooooooooooooooooi")
                        # if len(coor_slots_to_track)==0:
                        #     print("sssssssssssssssslotttttttttttttttt")

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort_list[i].increment_ages()
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            #------------------------------------------------------------------------------------------
            time_bef=time.time()
            paths = findPath(coor_car_to_track, list1, coor_slots_to_track, 360, 640, 40, 40)
            #print(":::::::::::::::::::::", car_to_track, slots_to_track)
            #print("++++++++++++++++++++++++++", paths)
            im0 = draw_path(paths, coor_car_to_track, list1, coor_slots_to_track, 360, 640,40, 40, im0)
            global time_find_path

            time_find_path += time.time()-time_bef

            #_------------------------------------------------------------------------------------------
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)
        
        # paths = findPath(car_to_track, list_poly, slots_to_track, 360, 640, 80, 80)
        # print("----------------------------------")
        # print(paths)
        # print('------------------------------------')
    time2=time.time()
    print('__________________________________________________________________')
    print("total_time: ", (time2-time1))
    print("time find path: ", time_find_path)
    print('__________________________________________________________________')
    

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_model)  # update model (to fix SourceChangeWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
