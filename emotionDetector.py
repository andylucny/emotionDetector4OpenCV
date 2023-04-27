# import the necessary packages
import numpy as np
import cv2 as cv

# Load Model
print("loading emotion detection model ...")
net2 = cv.dnn.readNet('mobilenet_7.pbtxt','mobilenet_7.pb')
labels = None
labelsFile = "labels.txt"
with open(labelsFile, 'rt') as f:
    labels = f.read().rstrip('\n').split('\n')
print("... face emotion model loaded")

def setTarget(gpu):
    if gpu:
        net2.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net2.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    else:
        net2.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net2.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

setTarget(False)

def detectEmotion(face):
    # transform the image to suitable input of the MobileNet DNN
    blob = cv.dnn.blobFromImage(face, 1.0, (224, 224), (123.68, 116.779, 103.939), True, False)
    # put the input to the network
    net2.setInput(blob)
    # launch the network and get the produced output
    scores = net2.forward()[0]
    # process the output typical for softmax classifier
    emotion = np.argmax(scores) # 0..6 see labels.txt
    return emotion
      
def displayEmotion(image, emotion):
    # display the detected emotion
    label = labels[emotion]
    print(label)
    cv.putText(image, label, (10,50), cv.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv.LINE_AA)

def main(filename):
    # load image
    image = cv.imread(filename)

    # detect emotion
    emotion = detectEmotion(image)

    # display emotion
    displayEmotion(image, emotion)
    
    # show the output image
    cv.imshow("Emotion", image)
    cv.waitKey(0)

    cv.destroyAllWindows()

if __name__ == '__main__':
    main('happy.png')
    main('fear.png')
    main('angry.png')
