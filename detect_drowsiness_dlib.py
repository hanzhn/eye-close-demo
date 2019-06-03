#/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'lumo_wang'

#import necessary packages
from  scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
# import playsound
import argparse
import imutils
import time
import cv2
import dlib

import matplotlib.pyplot as plt
fig_size = (10,2)
max_record_num = 300
refresh_per_record = 1



import collections
ear_queue=collections.deque()
num_queue=collections.deque()
ear_queue.append(0)
num_queue.append(0)

def sound_alarm(path):
    #paly an alarm sound
    #playsound.playsound(path)
    pass

lines = []
def plot():
    global lines
    print len(lines)
    if len(lines) > 0:
        ax.lines.remove(lines[0])
    lines = ax.plot(num_queue, ear_queue, 'r')
    plt.ylim(0,40)
    plt.xlim(0,max_record_num)
    fig.canvas.draw_idle()

def eye_aspect_ration(eye):
    #compute the euclidean distances between the two sets of
    #vertical eye landmarks (x,y)-coordinates
    A=dist.euclidean(eye[1],eye[5])
    B=dist.euclidean(eye[2],eye[4])

    #compute the euclidean distance between the horizontal
    #eye landmark (x,y)-coordinates
    C=dist.euclidean(eye[0],eye[3])

    #compute the eye aspect ratio
    ear=(A+B)/(2.0*C)
    return ear


#construct the argument parse and parse the arguments
ap=argparse.ArgumentParser()
# ap.add_argument("-p","--shape-predictor",type=str,default="shape_predictor_68_face_landmarks.dat",
#                 required=True,help="path to facial landmark predictor")
ap.add_argument("-p","--shape-predictor",type=str,default="shape_predictor_68_face_landmarks.dat",
                help="path to facial landmark predictor")
ap.add_argument("-a","--alarm",type=str,default="alarm.wav",
                help="path alarm .WAV file")
ap.add_argument("-w","--webcam",type=int,default=0,
                help="index of webcam on system")
args=vars(ap.parse_args())

EYE_AR_THRESH=0.20
EYE_AR_CONSE_FRAMES=25

COUNTER=0
ALARM_ON=False

#intialize dlib face dector HOG_based, create facial landmark predictor
print('[INFO] loading facial landmark predictor...')
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor(args['shape_predictor'])

(lStart,lEnd)=face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart,rEnd)=face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#start the video stream thread
print('[INFO] starting video stream thread...')
vs=VideoStream(src=args['webcam']).start()
time.sleep(1.0)

# plt.ion()
# fig, ax = plt.subplots(figsize=(10,2))
# plt.draw()

count = 0
while True:
    frame=vs.read()
    # cv2.imshow('big',frame)
    # key=cv2.waitKey(1)&0xFFF
    # if key ==ord('q'):
    #     break
    # continue
    frame=imutils.resize(frame,width=450)
    frame_ori = frame.copy()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects=detector(gray,0)
    print(rects)
    #if more than one face
    for rect in rects:
        print(rect)
        exit()
        shape=predictor(gray,rect)
        shape=face_utils.shape_to_np(shape)

        leftEye=shape[lStart:lEnd]
        rightEye=shape[rStart:rEnd]
        leftEAR=eye_aspect_ration(leftEye)
        rightEAR=eye_aspect_ration(rightEye)
        ear=(leftEAR+rightEAR)/2.0

        #visualize rectangle
        left = rect.left()
        top  = rect.top()
        right = rect.right()
        bottom = rect.bottom()
        # x, y, w, h = cv2.boundingRect(rect)
        # print x, y, w, h
        # cv2.rectangle(frame_ori, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        #visualize each eye regions
        leftEyeHull=cv2.convexHull(leftEye)
        rightEyeHull=cv2.convexHull(rightEye)
        cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #check drowsiness
        # ear_queue.append(ear*100)
        # count+=1
        # num_queue.append(count)
        # if len(ear_queue)>max_record_num:
        #     ear_queue.popleft()
        #     num_queue.pop()
        # print len(ear_queue), ear, count
        # if(count)%refresh_per_record==0:
        #     p=Thread(target=plot,args=())
        #     p.deamon=True
        #     p.start()

        if ear<EYE_AR_THRESH:
            COUNTER+=1
            if COUNTER>=EYE_AR_CONSE_FRAMES:
                if not ALARM_ON:
                    ALARM_ON=True
                    if args['alarm']!='':
                        # t=Thread(target=sound_alarm,args=(args['alarm'],))
                        # t.deamon=True
                        # t.start()
                        pass
                cv2.putText(frame,'Tired Alert!',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
        else:
            COUNTER=0
            ALARM_ON=False
        cv2.putText(frame,'EAR: {:.2f}'.format(ear),(300,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    #show the frame
    cv2.imshow('Frame',frame)
    #cv2.imshow('Orginal',frame_ori)
    key=cv2.waitKey(1)&0xFFF
    if key ==ord('q'):
        break
cv2.destroyAllWindows()
vs.stop()