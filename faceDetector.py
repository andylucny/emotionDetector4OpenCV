# import the necessary packages
import numpy as np
import cv2 as cv

# defining prototext and caffemodel paths
model_architecture = "deploy.prototxt"
model_weights = "res10_300x300_ssd_iter_140000.caffemodel"
height = 300
width = 300
mean = (104.0, 177.0, 123.0)
threshold = 0.5

# Load Model
print("loading face detection model ...")
net = cv.dnn.readNetFromCaffe(model_architecture,model_weights)
print("... face detection model loaded")

def setTarget(gpu):
    if gpu:
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    else:
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

setTarget(False)

def detectFaces(image):

    h, w = image.shape[:2] 

    # convert to RGB
    rgb = cv.cvtColor(image,cv.COLOR_BGR2RGB)

    # blob preparation
    blob = cv.dnn.blobFromImage(cv.resize(image,(width,height)),1.0,(width,height),mean)

    # passing blob through the network to detect and pridiction
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    rects = []
    for i in range(detections.shape[2]):
        # extract the confidence and prediction
        confidence = detections[0, 0, i, 2]
        # filter detections by confidence greater than the minimum
        if confidence > threshold:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype(np.int32)
            if startX >= 0 and startY >= 0 and endX < w and endY < h and startX < endX and startY < endY:
                rects.append((startX, startY, endX, endY, confidence))

    return rects
    
def displayFaces(image,rects,fps=None):
    result = np.copy(image)
    faces = []
    for rect in rects:
        startX, startY, endX, endY, confidence = rect
        cv.rectangle(result, (startX, startY), (endX, endY), (0, 0, 255), 2)
        text = "{:.2f}%".format(confidence * 100)
        cv.putText(result, text, (startX, startY-5), 0, 1.0, (0, 0, 255), 2)
        faces.append(np.copy(image[startY:endY,startX:endX,:]))
        if fps is not None:
            cv.putText(result, f"{fps:1.2f}", (8,20), 0, 1.0, (0, 255, 0), 2)

    return result, faces

if __name__ == '__main__':

    # load image
    image = cv.imread('person.jpg')

    # detect faces
    rects = detectFaces(image)

    # display
    result, _ = displayFaces(image, rects)
    
    # show the output image
    cv.imshow("Faces", cv.resize(result,(result.shape[0]//2,result.shape[1]//2)))
    cv.waitKey(0)
    
    cv.destroyAllWindows()

