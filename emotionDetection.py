import cv2 as cv
import numpy as np 
import time
from agentspace import space, Agent
from faceDetector import detectFaces, displayFaces, setTarget
from emotionDetector import detectEmotion, displayEmotion, setTarget as setTarget2
import sys
import os
import signal

# exit on ctrl-c
def signal_handler(signal, frame):
    os._exit(0)

signal.signal(signal.SIGINT, signal_handler)

def quit():
    os._exit(0)

# Initializing video capture agent
class CameraAgent(Agent):
    def init(self):
        camera = cv.VideoCapture(0)
        while True:
            _, frame = camera.read()
            space["bgr"] = frame
        
CameraAgent()
while space["bgr"] is None:
    time.sleep(0.5)

gpu = False
while True:
    frame = space["bgr"]

    t0 = time.time()
    
    rects = detectFaces(frame)
    
    t1 = time.time()
    fps = 1.0/(t1-t0)

    result, faces = displayFaces(frame, rects, fps)
    
    if len(faces) > 0:
        emotion = detectEmotion(faces[0])
        displayEmotion(result,emotion)

    cv.imshow('face',result)
    key = cv.waitKey(1) & 0xff
    if key == 27:
        break
    elif key == ord('g'):
        gpu = not gpu
        setTarget(gpu)
        setTarget2(gpu)

cv.destroyAllWindows()
if not sys.flags.interactive:
    quit()
