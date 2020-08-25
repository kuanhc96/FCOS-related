from fcos_core.config import cfg
from predictor import COCODemo

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import argparse
#parser.add_argument('--iou_thresh', type=float, default=0.1)
#args = parser.parse_args()

def read_json(filename):
    """
    Decodes a JSON file & returns its content.
    Raises:
        FileNotFoundError: file not found
        ValueError: failed to decode the JSON file
        TypeError: the type of decoded content differs from the expected (list of dictionaries)
    :param filename: [str] name of the JSON file
    :return: [list] list of the annotations
    """
    if not os.path.exists(filename):
        raise FileNotFoundError("File %s not found." % filename)
    try:
        with open(filename, 'r') as _f:
            _data = json.load(_f)
    except json.JSONDecodeError:
        raise ValueError(f"Failed to decode {filename}.")
    if not isinstance(_data, list):
        raise TypeError(f"Decoded content is {type(_data)}. Expected list.")
    if len(_data) > 0 and not isinstance(_data[0], dict):
        raise TypeError(f"Decoded content is {type(_data[0])}. Expected dict.")
    return _data

# get the ground-truth bbox coordinates from the json annotation files
def load_ground_truth_coordinates(view, filename): # filename format is 0000XXXX.png, XXXX is a timestamp
    annotations = read_json(filename)
    coordinates = list()
    for annotation in annotations:
        bbox = annotation['views'][view] # 0 - 6, representing C1 - C 7
        xmin = bbox['xmin']
        ymin = bbox['ymin']
        xmax = bbox['xmax']
        ymax = bbox['ymax']
        if (xmin, ymin, xmax, ymax) == (-1, -1, -1, -1): # person not present
            continue
        coordinate = list()
        coordinate.append((xmin, ymin))
        coordinate.append((xmax, ymax))
        coordinates.append(coordinate)

    return coordinates

# calculate the overlapping area between prediction bbox and ground truth bbox
def calc_overlap(gt_coordinate, p_coordinate):
    gt_lower = gt_coordinate[0]
    gt_upper = gt_coordinate[1]
    p_lower = p_coordinate[0]
    p_upper = p_coordinate[1]

    gt_area = (gt_upper[0] - gt_lower[0]) * (gt_upper[1] - gt_lower[1])
    p_area = (p_upper[0] - p_lower[0]) * (p_upper[1] - p_lower[1])
    '''
    def area(a, b):  # returns None if rectangles don't intersect
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
        if (dx>=0) and (dy>=0):
            return dx*dy
    '''

    dx = min(gt_upper[0], p_upper[0]) - max(gt_lower[0], p_lower[0])
    dy = min(gt_upper[1], p_upper[1]) - max(gt_lower[1], p_lower[1])
    if (dx >=0) and (dy>=0):
        return dx*dy, dx*dy / (gt_area + p_area - dx*dy)
    else:
        return 0, 0

# checks to see if a detection is TP or FP, and fills the TP and FP field in dataframe
def get_stats(ground_truth, iou_threshold, dataframe, imageID):

    # if a prediction is "seen" that means it is already matched to a ground-truth bbox and should not be reassigned to another ground-truth bbox
    # since, of course, one prediction is meant to only predict one groud-truth
    seen_predictions = list()

    # find matches for the ground-truth bboxes. A matching prediction bbox is one that is closest to it
    # if a ground truth bbox cannot find a match, that indicates an FN prediction occurred
    for gt_coordinate in ground_truth:
        # these values should change at the end of the ensuing for loop if a match is found
        max_area, max_area_percent = 0, 0
        max_area_prediction = None
        max_index = - 1

        for row in dataframe.itertuples(): # find the prediction corresponding to current ground-truth bbox
            # each "coordinate" represents a unique prediction (detection) that the algorithm made
            p_coordinate = row.coordinate
            index = row.Index # store the position of this prediction
            if p_coordinate not in seen_predictions: # the prediction is not matched with a ground-truth bbox
                
                area, area_percent = calc_overlap(gt_coordinate, p_coordinate) # check overlap between the prediction and ground-truth bbox
                if area > max_area: # the ground-truth and prediction pair with the greatest overlap (closest) compared to all other predictions is said to be a match
                    max_area = area
                    max_area_percent = area_percent
                    max_area_prediction = p_coordinate
                    max_index = index # get the index of the matching prediction
        #print("GT:", gt_coordinate, "P", max_area_prediction, "percent", max_area_percent)
        if max_area_prediction != None: # this means that the prediction was matched with a ground-truth bbox
            seen_predictions.append(max_area_prediction) # the prediction is now "seen" and should not be used again to make another match
            if max_area_percent >= iou_threshold: # this means that the prediction is valid
                dataframe.iloc[max_index, 3] += 1 # 3rd entry is TP
            else:
                dataframe.iloc[max_index, 4] += 1 # 4th entry is

def get_image_precision_recall_F1(image_dataframe, num_ground_truth):
    total_TP = np.sum(image_dataframe['TP'])
    total_FP = np.sum(image_dataframe['FP'])
    if total_TP == 0 and total_FP == 0:
        return 1, 0, 0
    elif total_TP == 0:
        return 1, 0, 0
    image_precision = total_TP / (total_FP + total_TP)
    image_recall = total_TP / num_ground_truth
    return image_precision, image_recall, 2 * image_precision * image_recall / (image_precision + image_recall)


### main ###
parser = argparse.ArgumentParser(description='run evaluation with iou threshold')
parser.add_argument('--iou_thresh', type=float, default=0.1)
args = parser.parse_args()
iou_threshold = args.iou_thresh
thresholds_for_classes = [
        0.3, 0.4928510785102844, 0.5040897727012634,
        0.4912887513637543, 0.5016880631446838, 0.5278812646865845,
        0.5351834893226624, 0.5003424882888794, 0.4955945909023285,
        0.43564629554748535, 0.6089804172515869, 0.666087806224823,
        0.5932040214538574, 0.48406165838241577, 0.4062422513961792,
        0.5571075081825256, 0.5671307444572449, 0.5268378257751465,
        0.5112953186035156, 0.4647842049598694, 0.5324517488479614,
        0.5795850157737732, 0.5152440071105957, 0.5280804634094238,
        0.4791383445262909, 0.5261335372924805, 0.4906163215637207,
        0.523737907409668, 0.47027698159217834, 0.5103300213813782,
        0.4645252823829651, 0.5384289026260376, 0.47796186804771423,
        0.4403403103351593, 0.5101461410522461, 0.5535093545913696,
        0.48472103476524353, 0.5006796717643738, 0.5485560894012451,
        0.4863888621330261, 0.5061569809913635, 0.5235867500305176,
        0.4745445251464844, 0.4652363359928131, 0.4162440598011017,
        0.5252017974853516, 0.42710989713668823, 0.4550687372684479,
        0.4943239390850067, 0.4810051918029785, 0.47629663348197937,
        0.46629616618156433, 0.4662836790084839, 0.4854755401611328,
        0.4156557023525238, 0.4763634502887726, 0.4724511504173279,
        0.4915047585964203, 0.5006274580955505, 0.5124194622039795,
        0.47004589438438416, 0.5374764204025269, 0.5876904129981995,
        0.49395060539245605, 0.5102297067642212, 0.46571290493011475,
        0.5164387822151184, 0.540651798248291, 0.5323763489723206,
        0.5048757195472717, 0.5302401781082153, 0.48333442211151123,
        0.5109739303588867, 0.4077408015727997, 0.5764586925506592,
        0.5109297037124634, 0.4685552418231964, 0.5148998498916626,
        0.4224434792995453, 0.4998510777950287
    ]

weight = "weights/FCOS_imprv_R_50_FPN_1x.pth"
cfg.merge_from_file("configs/fcos/fcos_imprv_R_50_FPN_1x.yaml")
cfg.merge_from_list(list())
cfg.MODEL.WEIGHT = weight
cfg.freeze()

for i in range(1):
    # image being examined, coordinate of a prediction (detection), confidnce score of a prediction, TP = 1 if prediction is tru positive, 0 otherwise, FP = 1 if prediction is false positive, 0 otherwise
    dataframe = pd.DataFrame(columns=("ImageID", "coordinate", "confidence", "TP", "FP"))

    # total number of ground truth bboxes = TP + FN (a ground-truth bbox is either detected or not detected), used in calculating recall
    total_gt_boxes = 0
    total_image_precision = 0
    total_recall = 0
    total_F1_score = 0
    total_images = 0

    # build the model from a config file and a checkpoint file
    coco_demo = COCODemo(
                            cfg,
                            confidence_thresholds_for_classes=thresholds_for_classes,
                            min_image_size=800
                        )
    for view in range(7): # examine 1 video at a time
        for time_stamp in range(0, 1996, 5): # examine each frame
            if int(time_stamp / 10) == 0: # single digit 000X
                stamp = "000" + str(time_stamp)
            elif int(time_stamp / 100) == 0: # two digits 00XX
                stamp = "00" + str(time_stamp)
            elif int(time_stamp / 1000) == 0: # three digits 0XXX
                stamp = "0" + str(time_stamp)
            else: # XXXX
                stamp = str(time_stamp)
            print("Processing C" + str(view + 1) + "0000" + stamp + ".png")
            
            # prepare object that handles inference plus adds predictions on top of image
            img = cv2.imread("../../Wildtrack_dataset/Image_subsets/C" + str(view + 1) + "/0000" + stamp + ".png")
            detections, confidence = coco_demo.get_person_detections(img)

            ground_truth_coordinates = load_ground_truth_coordinates(view, "../../Wildtrack_dataset/annotations_positions/0000" + stamp + ".json")

            num_ground_truth = len(ground_truth_coordinates)
            total_gt_boxes += num_ground_truth # add to number of ground-truth bboxes

            # create dataframe for each image, storing information regarding each prediction it contains 
            #Later appended to the "overall" dataframe
            image_dataframe = pd.DataFrame(columns=("ImageID", "coordinate", "confidence", "TP", "FP"))

            image_dataframe["ImageID"] = ["C"+str(view + 1) + "0000" + stamp] * len(detections)
            image_dataframe["coordinate"] = detections
            image_dataframe["confidence"] = confidence
            image_dataframe["TP"] = [0] * len(detections)
            image_dataframe["FP"] = [0] * len(detections)

            get_stats(ground_truth_coordinates, iou_threshold, image_dataframe, "C" + str(view+1) + "/0000" + stamp + ".png")
            precision, recall, F1 = get_image_precision_recall_F1(image_dataframe, num_ground_truth)
            total_image_precision += precision
            total_recall += recall
            total_F1_score += F1
            total_images += 1
            dataframe = pd.concat([dataframe, image_dataframe], ignore_index=True)

    print(dataframe)
    dataframe.sort_values(by=['confidence'], inplace=True,ascending=False, ignore_index=True) # sort by confidence

    # the next 4 attributes (columns) to be added to the overall dataframe
    acc_TP = list()
    acc_FP = list()
    precision = list()
    recall = list()

    current_acc_TP = 0 # accumulated number of TP seen up to current row
    current_acc_FP = 0 # accumulated number of FP seen up to current row

    for row in dataframe.itertuples(): # iterate over the rows to fill acc_TP, acc_FP, precision, and recall values
        current_acc_TP += dataframe.iloc[row.Index, 3] # 3rd column is TP
        current_acc_FP += dataframe.iloc[row.Index, 4] # 4th column is FP
        if current_acc_TP == 0 and current_acc_FP == 0: # edge case, when TP and FP are both 0, but FN is not, then let precision be 1
            current_precision = 1
        else:
            current_precision = current_acc_TP * 1.0 / (current_acc_TP + current_acc_FP) # precision up to current row
        precision.append(current_precision) # store current precision
        current_recall = current_acc_TP * 1.0 / total_gt_boxes # recall up to current row
        recall.append(current_recall) # store current recall
        acc_TP.append(current_acc_TP) # store current accumulated TP
        acc_FP.append(current_acc_FP) # store current accumulated FP

    # add columns into dataframe
    dataframe["acc_TP"] = acc_TP
    dataframe["acc_FP"] = acc_FP
    dataframe["precision"] = precision
    dataframe["recall"] = recall

    ##### Calculating AP ######

    # sort on recall values, increasing in value
    dataframe.sort_values(by=['recall'], inplace=True, ascending=True, ignore_index=True)
    print(dataframe)

    # max recall, beyond which precision will be 0
    max_recall = np.max(dataframe['recall'])

    # get only the precision and recall columns to calculate AP
    sub_dataframe = dataframe[['precision', 'recall']]
    for r in np.arange(0, 1.01, 0.1): # insert 11 points, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1 into sub_dataframe
        sub_dataframe = sub_dataframe.append({'precision':-1, 'recall':r}, ignore_index=True)

    sub_dataframe.sort_values(by=['recall'], inplace=True, ascending=False, ignore_index = True) # sort 11 points into sub_dataframe, let the recall to be in DESCENDING order so that interpolation can work backwards

    total_precision = 0
    current_max_precision = 0
    for row in sub_dataframe.itertuples():
        record = False
        if sub_dataframe.iloc[row.Index, 0] == -1: # one of the inserted values to be recorded for 11 point average
            record = True
        if sub_dataframe.iloc[row.Index, 1] > max_recall: # values that exceed the available max recall value should have precision = 0
            sub_dataframe.iloc[row.Index, 0] = 0
        else:
            if sub_dataframe.iloc[row.Index, 0] > current_max_precision: # update current max precision. THis is the value that in-between values will be interpolated to
                current_max_precision = sub_dataframe.iloc[row.Index, 0]
            else:
                sub_dataframe.iloc[row.Index, 0] = current_max_precision # interpolate
        if record:
            total_precision += sub_dataframe.iloc[row.Index, 0] # add the precision of the inserted value to the total

    print(sub_dataframe)
    # average total stats
    AP = total_precision / 11
    precision = total_image_precision / total_images
    recall = total_recall / total_images
    F1 = total_F1_score / total_images
    print("AP =", AP)
    print("Avg Precision =", precision)
    print("Avg Recall =", recall)
    print("Avg F1 score =", F1)

    try:
        f = open("stats.csv", "x") # file does not exist, error if file exists
        f.write("model,iou,AP,precision,recall,F1\n")
    except FileExistsError:
        f = open("stats.csv", "a") # file already exists, append to it
    f.write("FCOS,"+str(iou_threshold)+","+str(AP)+","+str(precision)+","+str(recall)+","+str(F1)+"\n")

