#############################################################################################################
##
##  Source code for testing
##
#############################################################################################################

import cv2
import json
import torch
#import sys
#sys.path.append('/home/kym/research/autonomous_car_vision/lanedection/code/')
#import util
import agent
import numpy as np
from copy import deepcopy
from data_loader import Generator
import time
from parameters import Parameters
import util

p = Parameters()

###############################################################
##
## Training
## 
###############################################################
def Testing():
    print('Testing')
    
    #########################################################################
    ## Get dataset
    #########################################################################
    print("Get dataset")
    loader = Generator()

    ##############################
    ## Get agent and model
    ##############################
    print('Get agent')
    if p.model_path == "":
        lane_agent = agent.Agent()
    else:
        lane_agent = agent.Agent()
        lane_agent.load_weights(640, "tensor(0.2298)")
    ##############################
    ## Check GPU
    ##############################
    print('Setup GPU mode')
    if torch.cuda.is_available():
        lane_agent.cuda()

    ##############################
    ## testing
    ##############################
    print('Testing loop')
    lane_agent.evaluate_mode()

    if p.mode == 0 : # check model with test data 
        for _, _, _, test_image in loader.Generate():
            _, _, ti = test(lane_agent, np.array([test_image]))
            cv2.imshow("test", ti[0])
            cv2.waitKey(0) 

    elif p.mode == 1: # check model with video
        cap = cv2.VideoCapture("/home/kym/research/autonomous_car_vision/lane_detection/code/Tusimple/SGPN_3216/test_data/LocalDataset_Day.mp4")
        while(cap.isOpened()):
            ret, frame = cap.read()
            prevTime = time.time()
            frame = cv2.resize(frame, (512,256))/255.0
            frame = np.rollaxis(frame, axis=2, start=0)
            _, _, ti = test(lane_agent, np.array([frame])) 
            curTime = time.time()
            sec = curTime - prevTime
            fps = 1/(sec)
            s = "FPS : "+ str(fps)
            cv2.putText(ti[0], s, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            cv2.imshow('frame',ti[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    elif p.mode == 2: # check model with a picture
        test_image = cv2.imread(p.test_root_url+"clips/0530/1492720840345996040_0/20.jpg")
        test_image = cv2.resize(test_image, (512,256))/255.0
        test_image = np.rollaxis(test_image, axis=2, start=0)
        _, _, ti = test(lane_agent, np.array([test_image]))
        cv2.imshow("test", ti[0])
        cv2.waitKey(0)   

    elif p.mode == 3: #evaluation
        print("evaluate")
        evaluation(loader, lane_agent)

############################################################################
## evaluate on the test dataset
############################################################################
def evaluation(loader, lane_agent, thresh = p.threshold_point, name = None):
    result_data = deepcopy(loader.test_data)
    for test_image, target_h, ratio_w, ratio_h, testset_index in loader.Generate_Test():
        x, y, _ = test(lane_agent, np.array([test_image]), thresh)
        x, y = util.convert_to_original_size(x[0], y[0], ratio_w, ratio_h)
        x, y = find_target(x, y, target_h, ratio_w, ratio_h)
        result_data = write_result_json(result_data, x, y, testset_index)
    if name == None:
        save_result(result_data, "test_result.json")
    else:
        save_result(result_data, name)

############################################################################
## linear interpolation for fixed y value on the test dataset
############################################################################
def find_target(x, y, target_h, ratio_w, ratio_h):
    # find exact points on target_h
    out_x = []
    out_y = []
    x_size = p.x_size/ratio_w
    y_size = p.y_size/ratio_h
    for i, j in zip(x,y):
        min_y = min(j)
        max_y = max(j)
        temp_x = []
        temp_y = []
        for h in target_h:
            temp_y.append(h)
            if h < min_y:
                temp_x.append(-2)
            elif min_y <= h and h <= max_y:
                for k in range(len(j)-1):
                    if j[k] >= h and h >= j[k+1]:
                        #linear regression
                        if i[k] < i[k+1]:
                            temp_x.append(int(i[k+1] - float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                        else:
                            temp_x.append(int(i[k+1] + float(abs(j[k+1] - h))*abs(i[k+1]-i[k])/abs(j[k+1]+0.0001 - j[k])))
                        break
            else:
                if i[0] < i[1]:
                    l = int(i[1] - float(-j[1] + h)*abs(i[1]-i[0])/abs(j[1]+0.0001 - j[0]))
                    if l > x_size or l < 0 :
                        temp_x.append(-2)
                    else:
                        temp_x.append(l)
                else:
                    l = int(i[1] + float(-j[1] + h)*abs(i[1]-i[0])/abs(j[1]+0.0001 - j[0]))
                    if l > x_size or l < 0 :
                        temp_x.append(-2)
                    else:
                        temp_x.append(l)
        out_x.append(temp_x)
        out_y.append(temp_y)
    
    return out_x, out_y

############################################################################
## write result
############################################################################
def write_result_json(result_data, x, y, testset_index):
    for i in x:
        result_data[testset_index]['lanes'].append(i)
        result_data[testset_index]['run_time'] = 1
    return result_data

############################################################################
## save result by json form
############################################################################
def save_result(result_data, fname):
    with open(fname, 'w') as make_file:
        for i in result_data:
            json.dump(i, make_file, separators=(',', ': '))
            make_file.write("\n")

############################################################################
## test on the input test image
############################################################################
def test(lane_agent, test_images, thresh = p.threshold_point):

    result = lane_agent.predict_lanes_test(test_images)
    confidences, offsets, instances = result[-1]
    
    num_batch = len(test_images)

    out_x = []
    out_y = []
    out_images = []

    for i in range(num_batch):
        # test on test data set
        image = deepcopy(test_images[i])
        image =  np.rollaxis(image, axis=2, start=0)
        image =  np.rollaxis(image, axis=2, start=0)*255.0
        image = image.astype(np.uint8).copy()

        confidence = confidences[i].view(p.grid_y, p.grid_x).cpu().data.numpy()

        offset = offsets[i].cpu().data.numpy()
        offset = np.rollaxis(offset, axis=2, start=0)
        offset = np.rollaxis(offset, axis=2, start=0)
        
        instance = instances[i].cpu().data.numpy()
        instance = np.rollaxis(instance, axis=2, start=0)
        instance = np.rollaxis(instance, axis=2, start=0)

        # generate point and cluster
        raw_x, raw_y = generate_result(confidence, offset, instance, thresh)

        # eliminate fewer points
        in_x, in_y = eliminate_fewer_points(raw_x, raw_y)
                
        # sort points along y 
        in_x, in_y = util.sort_along_y(in_x, in_y)  
        in_x, in_y = eliminate_out(in_x, in_y, confidence, deepcopy(image))
        in_x, in_y = util.sort_along_y(in_x, in_y)
        in_x, in_y = eliminate_fewer_points(in_x, in_y)

        result_image = util.draw_points(in_x, in_y, deepcopy(image))

        out_x.append(in_x)
        out_y.append(in_y)
        out_images.append(result_image)

    return out_x, out_y,  out_images

############################################################################
## post processing for eliminating outliers
############################################################################
def eliminate_out(sorted_x, sorted_y, confidence, image = None):
    out_x = []
    out_y = []

    for lane_x, lane_y in zip(sorted_x, sorted_y):

        lane_x_along_y = np.array(deepcopy(lane_x))
        lane_y_along_y = np.array(deepcopy(lane_y))

        ind = np.argsort(lane_x_along_y, axis=0)
        lane_x_along_x = np.take_along_axis(lane_x_along_y, ind, axis=0)
        lane_y_along_x = np.take_along_axis(lane_y_along_y, ind, axis=0)
        
        if lane_y_along_x[0] > lane_y_along_x[-1]: #if y of left-end point is higher than right-end
            starting_points = [(lane_x_along_y[0], lane_y_along_y[0]), (lane_x_along_y[1], lane_y_along_y[1]), (lane_x_along_y[2], lane_y_along_y[2]),
                                (lane_x_along_x[0], lane_y_along_x[0]), (lane_x_along_x[1], lane_y_along_x[1]), (lane_x_along_x[2], lane_y_along_x[2])] # some low y, some left/right x
        else:
            starting_points = [(lane_x_along_y[0], lane_y_along_y[0]), (lane_x_along_y[1], lane_y_along_y[1]), (lane_x_along_y[2], lane_y_along_y[2]),
                                (lane_x_along_x[-1], lane_y_along_x[-1]), (lane_x_along_x[-2], lane_y_along_x[-2]), (lane_x_along_x[-3], lane_y_along_x[-3])] # some low y, some left/right x            
    
        temp_x = []
        temp_y = []
        for start_point in starting_points:
            temp_lane_x, temp_lane_y = generate_cluster(start_point, lane_x, lane_y, image)
            temp_x.append(temp_lane_x)
            temp_y.append(temp_lane_y)
        
        max_lenght_x = None
        max_lenght_y = None
        max_lenght = 0
        for i, j in zip(temp_x, temp_y):
            if len(i) > max_lenght:
                max_lenght = len(i)
                max_lenght_x = i
                max_lenght_y = j
        out_x.append(max_lenght_x)
        out_y.append(max_lenght_y)

    return out_x, out_y

############################################################################
## generate cluster
############################################################################
def generate_cluster(start_point, lane_x, lane_y, image = None):
    cluster_x = [start_point[0]]
    cluster_y = [start_point[1]]

    point = start_point
    while True:
        points = util.get_closest_upper_point(lane_x, lane_y, point, 3)
         
        max_num = -1
        max_point = None

        if len(points) == 0:
            break
        if len(points) < 3:
            for i in points: 
                cluster_x.append(i[0])
                cluster_y.append(i[1])                
            break
        for i in points: 
            num, shortest = util.get_num_along_point(lane_x, lane_y, point, i, image)
            if max_num < num:
                max_num = num
                max_point = i

        total_remain = len(np.array(lane_y)[np.array(lane_y) < point[1]])
        cluster_x.append(max_point[0])
        cluster_y.append(max_point[1])
        point = max_point
        
        if len(points) == 1 or max_num < total_remain/5:
            break

    return cluster_x, cluster_y

############################################################################
## remove same value on the prediction results
############################################################################
def remove_same_point(x, y):
    out_x = []
    out_y = []
    for lane_x, lane_y in zip(x, y):
        temp_x = []
        temp_y = []
        for i in range(len(lane_x)):
            if len(temp_x) == 0 :
                temp_x.append(lane_x[i])
                temp_y.append(lane_y[i])
            else:
                if temp_x[-1] == lane_x[i] and temp_y[-1] == lane_y[i]:
                    continue
                else:
                    temp_x.append(lane_x[i])
                    temp_y.append(lane_y[i])     
        out_x.append(temp_x)  
        out_y.append(temp_y)  
    return out_x, out_y

############################################################################
## eliminate result that has fewer points than threshold
############################################################################
def eliminate_fewer_points(x, y):
    # eliminate fewer points
    out_x = []
    out_y = []
    for i, j in zip(x, y):
        if len(i)>2:
            out_x.append(i)
            out_y.append(j)     
    return out_x, out_y   

############################################################################
## generate raw output
############################################################################
def generate_result(confidance, offsets,instance, thresh):

    mask = confidance > thresh
    #print(mask)

    grid = p.grid_location[mask]
    offset = offsets[mask]
    feature = instance[mask]

    lane_feature = []
    x = []
    y = []
    for i in range(len(grid)):
        if (np.sum(feature[i]**2))>=0:
            point_x = int((offset[i][0]+grid[i][0])*p.resize_ratio)
            point_y = int((offset[i][1]+grid[i][1])*p.resize_ratio)
            if point_x > p.x_size or point_x < 0 or point_y > p.y_size or point_y < 0:
                continue
            if len(lane_feature) == 0:
                lane_feature.append(feature[i])
                x.append([])
                x[0].append(point_x)
                y.append([])
                y[0].append(point_y)
            else:
                flag = 0
                index = 0
                for feature_idx, j in enumerate(lane_feature):
                    index += 1
                    if index >= 12:
                        index = 12
                    if np.linalg.norm((feature[i] - j)**2) <= p.threshold_instance:
                        lane_feature[feature_idx] = (j*len(x[index-1]) + feature[i])/(len(x[index-1])+1)
                        x[index-1].append(point_x)
                        y[index-1].append(point_y)
                        flag = 1
                        break
                if flag == 0:
                    lane_feature.append(feature[i])
                    x.append([])
                    x[index].append(point_x) 
                    y.append([])
                    y[index].append(point_y)
                
    return x, y

if __name__ == '__main__':
    Testing()
