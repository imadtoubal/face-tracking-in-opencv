import cv2
import numpy as np
import math
import argparse
import os
from tqdm import tqdm


def lerp(a, b, p):
    return math.floor(a + (b - a) * p)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
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
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', help="Path to the input videos directory.")
    parser.add_argument('--size', help="Output size.")
    args = parser.parse_args()

    out_size = int(args.size)
    threshold = 10
    margin = 80

    video_names = [vpath for vpath in os.listdir(
        args.inpath) if vpath[-4:] == '.mp4']

    for video_name in tqdm(video_names, total=len(video_names)):
        path_to_video = os.path.join(args.inpath, video_name)
        output_path = os.path.join(
            args.inpath, f'{video_name[:-4]}_{str(out_size)}.avi')

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), 30, (out_size, out_size))

        cap = cv2.VideoCapture(path_to_video)
        face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')

        first_frame = True
        f = 0
        x = -1

        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break
            # Our operations on the frame come here
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            gray_re = image_resize(gray, width=400)

            height, width = gray.shape
            height_re, width_re = gray_re.shape

            ratio = height / height_re

            # Detect the faces
            faces = face_cascade.detectMultiScale(gray_re, 1.1, 5)

            # Draw the rectangle around each face
            maxArea = 0
            minDist = np.inf

            for (_x, _y, _w, _h) in faces:
                # cv2.rectangle(gray, (_x, _y), (x+fw, y+fh), (255, 0, 0), 2)
                if first_frame and _w*_h > maxArea:
                    x, y, w, h = _x, _y, _w, _h

                elif (_x - _xp) ** 2 + (_y - _yp) ** 2 < minDist:
                    x, y, w, h = _x, _y, _w, _h
                    minDist = (_x - _xp) ** 2 + (_y - _yp) ** 2

                maxArea = w*h
                _xp = x
                _yp = y
            # If one or more faces are found, draw a rectangle around the
            # largest face present in the picture
            if maxArea > 0:
                if minDist < threshold:
                    # add margin
                    x = max(math.floor(x * ratio - margin), 0)
                    y = max(math.floor(y * ratio - margin), 0)
                if first_frame:
                    fw = math.floor(w * ratio + 2 * margin)
                    fh = math.floor(h * ratio + 2 * margin)

            # get rectangle
            if not first_frame:
                lerp_p = .3
                if minDist < threshold:
                    x, y = lerp(xp, x, lerp_p), lerp(yp, y, lerp_p)
                else:
                    x, y = xp, yp

            if x >= 0:
                patch = frame[y:y+fh, x:x+fw]
                # Convert patch to feature vector

                xp, yp = x, y

            # Display the resulting frame
            try:
                # cv2.imwrite('output_{}.png'.format(f), patch)
                f = f + 1
                frameout = image_resize(patch, width=out_size, height=out_size)
                out.write(frameout)

            except:
                print('output/finding face')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # check if faces were found
            if (x >= 0):
                first_frame = False

        cap.release()
        out.release()
        cv2.destroyAllWindows()
