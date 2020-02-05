import cv2
import numpy as np 
import math

path_to_video = 'video.mp4'
margin = 50


# Helper function
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


cap = cv2.VideoCapture(path_to_video)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

first_frame = True
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_re = image_resize(gray, width=400)
    
    height, width = gray.shape
    height_re, width_re = gray_re.shape

    ratio = height / height_re

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray_re, 1.2, 5)

    # Draw the rectangle around each face
    maxArea = 0
    minDist = np.inf
    for (_x, _y, _w, _h) in faces:

        if  first_frame and _w*_h > maxArea:
            x, y, w, h = _x, _y, _w, _h
            first_frame = False

        elif (_x - xp) ^ 2 + (_y - yp) ^ 2 < minDist:
            x, y, w, h = _x, _y, _w, _h
            minDist = (_x - xp) ^ 2 + (_y - yp) ^ 2  
        
        maxArea = w*h
        xp = x
        yp = y
        

    #If one or more faces are found, draw a rectangle around the
    #largest face present in the picture
    if maxArea > 0 :
        x = max(math.floor(x * ratio - margin), 0)
        y = max(math.floor(y * ratio - margin), 0)
        w = math.floor(w * ratio + 2 * margin)
        h = math.floor(h * ratio + 2 * margin) 
        imtoshow = gray[y:y+h, x:x+w]
        # cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        

    # Display the resulting frame
    cv2.imshow('frame', image_resize(imtoshow, height=400, width=400) )
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()