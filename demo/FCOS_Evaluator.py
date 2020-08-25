from Base_Evaluator import BaseEvaluator
from fcos_core.config import cfg
from predictor import COCODemo
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import argparse
import threading
import time
class FCOSEvaluator(BaseEvaluator):
    def __init__(self, weights, iou_thresh=0.1, detector_name="FCOS", config=None):
        self.THRESHOLDS_FOR_CLASSES = [
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
        print(config)
        super(FCOSEvaluator, self).__init__(iou_thresh, detector_name, weights, config=config)

    def read_json(self, filename):
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
    def load_ground_truth_coordinates(self, filename, view=None): # filename format is 0000XXXX.png, XXXX is a timestamp
        if view == None:
            raise TypeError("view cannot be None")
        annotations = self.read_json(filename)
        coordinates = list()
        for annotation in annotations:
            bbox = annotation['views'][view-1] # 0 - 6, representing C1 - C 7
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

    def prepare_detector(self):
        # Load model
        cfg.merge_from_file(self.config)
        cfg.merge_from_list(list())
        cfg.MODEL.WEIGHT = self.weights
        cfg.freeze()
        self.detector = COCODemo(
                cfg, 
                confidence_thresholds_for_classes=self.THRESHOLDS_FOR_CLASSES, 
                min_image_size=800)

    def load_detections_and_confidence(self, image):
        #self.prepare_detector()
        detections, confidence = self.detector.get_person_detections(image)
        return detections, confidence

    def evaluate_one_set(self, view):
        # for view in range(7): # examine 1 video at a time
        for time_stamp in range(0, 1996, 5): # examine each frame
            if int(time_stamp / 10) == 0: # single digit 000X
                stamp = "000" + str(time_stamp)
            elif int(time_stamp / 100) == 0: # two digits 00XX
                stamp = "00" + str(time_stamp)
            elif int(time_stamp / 1000) == 0: # three digits 0XXX
                stamp = "0" + str(time_stamp)
            else: # XXXX
                stamp = str(time_stamp)
            print("Processing C" + str(view) + "0000" + stamp + ".png")
            img = cv2.imread("../../Wildtrack_dataset/Image_subsets/C" + str(view) + "/0000" + stamp + ".png")
            start = time.time()
            detections, confidence = self.load_detections_and_confidence(img)
            end = time.time()
            image_time_elapsed = end-start
            image_num_detections = len(detections)
            self.imageIDs.append("C" + str(view) + "0000" + stamp + ".png")
            self.num_detections.append(image_num_detections)
            self.times.append(image_time_elapsed)
            ground_truth_coordinates = self.load_ground_truth_coordinates("../../Wildtrack_dataset/annotations_positions/0000" + stamp + ".json", view=view)
            self.classify_detections(detections, confidence, ground_truth_coordinates, "C"+str(view)+"0000"+stamp+".png")
        #self.calculate_final_stats()

    def evaluate(self):
        #x = threading.Thread(target=thread_function, args=(1,))
        self.prepare_detector()
        threads = list()
        for view in range(1, 8): # examine 1 video at a time
            #print("view:", view)
            #t = threading.Thread(target=self.evaluate_one_set, args=(view,))
            #t.start()
            #threads.append(t)
            self.evaluate_one_set(view)
        #for thread in threads:
            #thread.join()
        self.timer_dataframe["ImageID"] = self.imageIDs
        self.timer_dataframe["num_detections"] = self.num_detections
        self.timer_dataframe["time"] = self.times
        plt.plot(self.num_detections, self.times, 'ro')
        plt.title("num_detections vs time")
        plt.xlabel("number of detections")
        plt.ylabel("time elapsed")
        plt.savefig("times plot.png", format='png')
        self.timer_dataframe.to_csv("times.csv",index=False)
        self.calculate_final_stats()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run evaluation with iou threshold')
    parser.add_argument('--iou_thresh', type=float, default=0.1)
    args = parser.parse_args()
    evaluator = FCOSEvaluator( "weights/FCOS_imprv_R_50_FPN_1x.pth", iou_thresh=args.iou_thresh, detector_name="FCOS", config="configs/fcos/fcos_imprv_R_50_FPN_1x.yaml")
    evaluator.evaluate()

