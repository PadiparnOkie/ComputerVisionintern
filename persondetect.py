import numpy as np
import datetime
import cv2
from ultralytics import YOLO

from deep_sort.deep_sort.tracker import Tracker
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.tools import generate_detections as gdet

from helper import create_video_writer


# define some parameters
conf_threshold = 0.5
max_cosine_distance = 0.5
nn_budget = None

counter = []
current_count = int(0)

# Initialize the video capture and the video writer objects
cap = cv2.VideoCapture('/Users/okiepadiparn/Documents/persondetection/people.mp4')


# Initialize the YOLOv8 model using the default weights
model = YOLO("yolov8n.pt")


# Initialize the deep sort tracker
model_filename = "/Users/okiepadiparn/Documents/persondetection/mars-small128 (1).pb"
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric, max_age = 100)

# load the COCO class labels the YOLO model was trained on
classes_path = "/Users/okiepadiparn/Documents/persondetection/coco.names"
with open(classes_path, "r") as f:
    class_names = f.read().strip().split("\n")

# create a list of random colors to represent each class
np.random.seed(42)  # to get the same colors
colors = np.random.randint(0, 255, size=(len(class_names), 3))  # (80, 3)

from _collections import deque
pts = [deque(maxlen=30) for _ in range(1000)]

# loop over the frames
while True:
    # starter time to computer the fps
    start = datetime.datetime.now()
    ret, frame = cap.read()

    # if there is no frame, we have reached the end of the video
    if not ret:
        print("End of the video file...")
        break


    ## run the YOLO model on the frame
    results = model(frame)

    # loop over the results
    for result in results:
        # initialize the list of bounding boxes, confidences, and class IDs
        bboxes = []
        confidences = []
        class_ids = []
        appearance_features = []

        # loop over the detections
        for data in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = data
            x = int(x1)
            y = int(y1)
            w = int(x2) - int(x1)
            h = int(y2) - int(y1)
            class_id = int(class_id)
            
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if class_names[class_id] != 'person':
                continue

            # filter out weak predictions by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > conf_threshold:
                bboxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)
                
                
    ############################################################
    ### Track the objects in the frame using DeepSort        ###
    ############################################################

    # get the names of the detected objects
    names = [class_names[class_id] for class_id in class_ids]


    # get the features of the detected objects
    features = encoder(frame, bboxes)

    # convert the detections to deep sort format
    dets = []

    for bbox, conf, class_name, feature in zip(bboxes, confidences, names, features):
        dets.append(Detection(bbox, conf, class_name, feature))

    # run the tracker on the detections
    tracker.predict()
    tracker.update(dets)

    # loop over the tracked objects
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        # get the bounding box of the object, the name
        # of the object, and the track id
        bbox = track.to_tlbr()
        track_id = track.track_id
        class_name = track.get_class()
        # convert the bounding box to integers
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # get the color associated with the class name

        color = colors[track_id]
        B, G, R = int(color[0]), int(color[1]), int(color[2])

        # draw the bounding box of the object, the name
        # of the predicted object, and the track id
        text = f"{track_id} - {class_name} ({confidence:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)
        cv2.rectangle(frame, (x1 - 1, y1 - 20),
                      (x1 + len(text) * 12, y1), (B, G, R), -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        #Optical flow
        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        pts[track.track_id].append(center)
        trcolor = colors[track_id]

        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                continue
            thickness = int(np.sqrt(64/float(j+1))*2)
            cv2.line(frame, (pts[track.track_id][j-1]), (pts[track.track_id][j]), trcolor.tolist(), thickness)

        center_y = int(((bbox[1])+(bbox[3]))/2)

    ############################################################
    ### Some post-processing to display the results          ###
    ############################################################

    # end time to compute the fps
    end = datetime.datetime.now()
    # calculate the frame per second and draw it on the frame

    if class_name == 'person':
                counter.append(int(track.track_id))
                current_count += 1
    
    total_count = len(set(counter))
    cv2.putText(frame, "Current People Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)
    cv2.putText(frame, "Total People Count: " + str(total_count), (0, 130), 0, 1, (0, 0, 255), 2)

    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50), 0, 1, (0, 0, 255), 2)
    cv2.imshow("Output", frame)
    # write the frame to disk

    if cv2.waitKey(1)&0xFF == 27:
        break

# release the video capture, video writer, and close all windows
cap.release()
cv2.destroyAllWindows()