#/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'lumo_wang'

#import necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
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
max_record_num = 200
refresh_per_record = 1

import collections
ear_queue=collections.deque()
num_queue=collections.deque()
status_queue=collections.deque()
ear_queue.append(0)
num_queue.append(0)
status_queue.append(0)

def sound_alarm(path):
    #paly an alarm sound
    #playsound.playsound(path)
    pass

import math
def softmax0(a, b):
    a = math.exp(a)
    b = math.exp(b)
    return a/(a+b)

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

EYE_AR_THRESH=0.10
EYE_AR_CONSE_FRAMES=25

COUNTER=0
ALARM_ON=False

#intialize dlib face dector HOG_based, create facial landmark predictor
print('[INFO] loading facial landmark predictor...')
detector=dlib.get_frontal_face_detector()
from model import face_cls
predictor=face_cls.Model()

#start the video stream thread
print('[INFO] starting video stream thread...')
vs=VideoStream(src=args['webcam']).start()
time.sleep(1.0)

lines = []
def plot():
    global lines
    # print len(lines)
    # if len(lines) > 0:
    #     ax.lines.remove(lines[0])
    lines = ax.plot(num_queue, ear_queue, color='red')
    thresh_line = [EYE_AR_THRESH*100 for i in range(len(ear_queue))]
    lines = ax.plot(num_queue, thresh_line, color='green', linewidth=0.5, linestyle='-')
    lines = ax.plot(num_queue, status_queue, color='blue')

    plt.ylim(-10,110)
    # print('plot ',num_queue[0])
    beginning = max(0,num_queue[-1]-max_record_num)
    plt.xlim(beginning,beginning+max_record_num)
    # plt.xlim(0,max_record_num)
    fig.canvas.draw_idle()

# plt.ion()
# fig, ax = plt.subplots(figsize=(10,2))
# plt.draw()

count = 0
while True:
    frame=vs.read()
    count+=1
    if count%2 != 0:
        continue
    # cv2.waitKey(0)
    # frame = cv2.imread('2.jpg')
    frame=imutils.resize(frame,width=450)
    height, width,_=list(frame.shape)
    frame_ori = frame.copy()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects=detector(gray,0)
    # print(rects)
    #if more than one face
    bigest_idx = -1
    bigest_size = 0
    for i, rect in enumerate(rects):
        size = rect.bottom()-rect.top()+rect.right()-rect.left()
        if size > bigest_size:
            bigest_idx = i
            bigest_size = size
    if bigest_idx < 0:
        #show the frame
        cv2.imshow('Frame',frame)
        # cv2.imshow('Orginal',frame_ori)
        key=cv2.waitKey(1)&0xFFF
        if key ==ord('q'):
            break
        continue

    rect = rects[bigest_idx]
    # print(rect)
    #visualize rectangle
    h = rect.bottom()-rect.top()
    top = int(rect.top()-0.3*h)
    left = max(rect.left(),0)
    # top  = max(rect.top(),0)
    top  = max(top,0)
    right = min(rect.right(),width)
    bottom = min(rect.bottom(),height)
    
    
    face=frame_ori[top:bottom, left:right]
    face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)

    res = predictor.process(face)
    res=res[0][0]
    # print(res)
    # ear = int(0.7*res[0] > 0.3*res[1])
    ear = softmax0(res[0],res[1])
    ear1 = softmax0(res[0],res[1])
    print('ear:%0.2f'%ear, 'ear1:%0.2f'%ear1)
    

    #check drowsiness
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

    status = int(COUNTER<=0)
    if ALARM_ON:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    elif status:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    else:
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

    # ear_queue.append(ear*100)
    # num_queue.append(count)
    # status_queue.append(status*100)
    # if len(ear_queue)>max_record_num:
    #     ear_queue.popleft()
    #     num_queue.popleft()
    #     status_queue.popleft()
    #     # num_queue.pop()
    # print len(ear_queue), ear, count
    # if(count)%refresh_per_record==0:
    #     p=Thread(target=plot,args=())
    #     p.deamon=True
    #     p.start()

    # cv2.putText(frame,'Status:{}'.format(status),(50,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    cv2.putText(frame,'Prob: {:.2f}'.format(ear),(300,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    #show the frame
    cv2.imshow('Frame',frame)
    # cv2.imshow('Orginal',frame_ori)
    key=cv2.waitKey(1)&0xFFF
    if key ==ord('q'):
        break
cv2.destroyAllWindows()
vs.stop()