from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0 # initialize unique object ID
        self.objects = OrderedDict() # ID -> centroid
        self.disappeared = OrderedDict() # ID -> number of frames disappeared
        self.maxDisappeared = maxDisappeared # max allowable frames an object can be absent

    def register(self, centroid):
        # use next available ID to store the centroid
        self.objects[self.nextObjectID] = centroid # assign ID to centroid 
        self.disappeared[self.nextObjectID] = 0 # new ID has disappeared 0 frames
        self.nextObjectID += 1 # get next available ID

    def deregister(self, objectID):
        # remove objectID from both dictionaries
        del self.objects[objectIS]
        del self.disappeared[objectID]

    def update(self, rects):
        # rects = input bbox from object detection
        # rects format: (x1, y1, x2, y2)
        if len(rects) == 0: # check if rects is empty -- all objects disappeared in this frame
            for objectID in list(self.disappeared.keys()): # loop over all objectIDs
                self.disappeared[objectID] += 1 # all items have disappeared in this frame
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID) # remove objectID that exceeds maxDisappeared
            return self.objects # nothing else to do if input is empty
        
        inputCentroids = np.zeros((len(rects), 2), dtype="int") # array of input centroids

        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            # use bbox coordinates to derive centroid
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            inputCentroids[i] = (cX, cY) # put cX, cY into inputCentroids

        if len(self.objects) == 0: # no objects being tracked, just register all the centroids
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else: # try to match centroids to existing object centroids
            # get set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute distance between each pair of object centroids and input centroids
            # each row stores distances between one of the objectCentroids and all inputCentroids
            # the smallest value in each row is the shortest distance between the object centroid and
            # a respective inputCentroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # 1: find smallest value in each row
            # 2: sort the row indices based on their minimum values so that the row with the smallest value is at the
            # front of the index list
            rows = D.min(axis=1).argsort()
            # perform similar process on columns by finding smallest value in each column and then sorting using the
            # previously computed row index list
            cols = D.argmin(axis=1)[rows]
            
            # keep track of which of the rows and column indices have already been examined
            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col] 
                self.disappeared[objectID] = 0 # since object appeared, reset

                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indices
                for row in unusedRows:
                    # grab the object ID for the corresponging row index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    # check if number of consecutive frames the object has been marked "disappeared" for warrants
                    # deregistring
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
                else:
                    # otherwise, if the number of input centroids is greater than the number of existing object
                    # centroids we need to register each new input centroid as a trackable object
                    for col in unusedCols:
                        self.register(inputCentroids[col])
            return self.objects


