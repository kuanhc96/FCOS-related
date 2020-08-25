# USAGE
# python FCOS_L2_tracker.py --model FCOS_imprv_R_50_FPN_1x.pth --video path/to/video.mp4 --config fcos_imprv_R_50_FPN_1x.yaml

# import the necessary packages
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import FileVideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from FCOS_Evaluator import FCOSEvaluator
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to pre-trained model")
ap.add_argument("-v", "--video", required=True,
        help="path to video file")
ap.add_argument("-c", "--config", required=True,
        help="path to configuration file")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
H, W = None, None

fcos_evaluator = FCOSEvaluator(0.1, "FCOS", args['model'], config=args['config'])
fcos_evaluator.prepare_detector()

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")

vs = FileVideoStream(args["video"]).start()
time.sleep(1.0)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
original_fps = vs.stream.get(cv2.CAP_PROP_FPS)
video_writer = None
frame_num = 0
# loop over the frames from the video stream
while vs.more():
    # read the next frame from the video stream and resize it
    frame = vs.read()
    if frame is None:
        print("finished processing")
        break
        
    print("processing frame", frame_num)
    frame = imutils.resize(frame, height=800)
    detections, confidence = fcos_evaluator.load_detections_and_confidence(frame)
    #print(detections)
    #print(confidence)
    # if the frame dimensions are None, grab them
    if W is None or H is None:
        H, W = frame.shape[:2]
        video_writer = cv2.VideoWriter("tracked_"+args["video"], fourcc, original_fps, (W, H))

    rects = []

    # loop over the detections
    for i in range(0, len(detections)):
        box = detections[i]
        box = np.array([box[0][0], box[0][1], box[1][0], box[1][1]])
        rects.append(box.astype("int"))

        # draw a bounding box surrounding the object so we can
        # visualize it
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    video_writer.write(frame)
    frame_num += 1


# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
video_writer.release()
