import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np


def faceBox(faceNet,frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.85:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 1)
    return frame, bboxs


emotion_model = load_model('/Users/okiepadiparn/Documents/FaceDetectionProj/train_emotion.h5')

faceProto = "/Users/okiepadiparn/Documents/Facedetect2/opencv_face_detector.pbtxt"
faceModel = "/Users/okiepadiparn/Documents/Facedetect2/opencv_face_detector_uint8.pb"

ageProto = "/Users/okiepadiparn/Documents/Facedetect2/age_deploy.prototxt"
ageModel = "/Users/okiepadiparn/Documents/Facedetect2/age_net.caffemodel"

genderModel = load_model('/Users/okiepadiparn/Documents/FaceDetectionProj/gender_model.h5')

faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
emotion_label = ['Angry','Disgust', 'Fear', 'Happy', 'Neutral','Sad','Surprise']

video=cv2.VideoCapture(0)

padding=20

while True:
    ret,frame=video.read()
    frame,bboxs=faceBox(faceNet,frame)
    for bbox in bboxs:
        # face=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]

        roi_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        roi_color = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        roi_color = cv2.resize(roi_color, (200, 200))

        gender_predict = genderModel.predict(np.array(roi_color).reshape(-1,200,200,3))
        gender_predict = (gender_predict>= 0.5).astype(int)[:,0]
        gender_label=genderList[gender_predict[0]] 

    

        # Perform emotion prediction on the ROI
        preds = emotion_model.predict(roi)[0]
        emotion = emotion_label[preds.argmax()]


        label="{},{},{}".format(gender_label, age, emotion)
        cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,0,255), -1) 
        cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2,cv2.LINE_AA)

    cv2.imshow("img", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

video.release()
cv2.destroyAllWindows()