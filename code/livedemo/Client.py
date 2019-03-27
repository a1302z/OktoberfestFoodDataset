"""
This is the client for the live demo.
It takes a picture with the android smartphone, sends this picture to the server and displays the result.
"""

import os
import socket
import subprocess
import cv2
import time
import numpy as np
from Utils import send_np_array, receive_np_array
import argparse as ap


parser = ap.ArgumentParser(description='Live-Demo Client')
parser.add_argument('--ip', type=str, default='localhost')
parser.add_argument('--port', type=int, default=14441)
parser.add_argument('--shutter_pos', type=str, default='540,1800',
                    help='The position of the button to take a photo in the camera app. Separated by an "," (e.g. 540,1800)')

category_names = ['Bier', 'Bier Maß', 'Weißbier', 'Cola', 'Wasser', 'Curry-Wurst', 'Weißwein',
                   'A-Schorle', 'Jägermeister', 'Pommes', 'Burger', 'Williamsbirne', 'Alm-Breze', 'Brotzeitkorb',
                   'Käsespätzle']


def show_image(img, bb, classes, scores, thresh=.4):
    """
    This function display the image
    img: the image
    bb: the bounding boxes
    classes: the class predictions
    scores: the confidence scores
    thresh: the threshold to filter out bounding boxes
    """
    h, w = img.shape[:2]
    # we do not want to bounding boxes on to of each other, since this doesn't look nice
    mask = non_maximum_supression(bb, scores)
    bxs = bb[mask]
    clss = classes[mask]
    scr = scores[mask]

    # draw the bounding boxes on to the image
    for j in range(len(bxs)):
        if scr[j] < thresh:
            continue
        x1, y1 = int(bxs[j, 1] * h), int(bxs[j, 0] * h)
        x2, y2 = int(bxs[j, 3] * h), int(bxs[j, 2] * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.addText(img, '%s: %.0f%%' % (category_names[int(clss[j])], scr[j] * 100), (x1, y1), 'Consolas', 14)

    # show the image with the bounding boxes
    cv2.imshow(window_name, img)
    return img


def iou(a, b):
    """
    a: bounding box a
    b: bounding box b
    returns: intersection over union between the two bounding boxes
    """
    y1 = max(a[0], b[0])
    x1 = max(a[1], b[1])
    y2 = min(a[2], b[2])
    x2 = min(a[3], b[3])

    iu = max(x2 - x1, 0) * max(y2 - y1, 0)

    iab = (a[3] - a[1]) * (a[2] - a[0]) + (b[3] - b[1]) * (b[2] - b[0])

    return iu / (iab - iu)


def non_maximum_supression(b, s, iou_thresh=.9, c_thresh=0.):  # b and s are sorted by s desc
    """
    perform non-maximum suppression
    b: the bounding boxes sorted by s desc
    s: the confidence scores sorted desc
    iou_thresh: the iou threshold at which two bounding boxes overlap
    c_thresh: a threshold to filter out prediction with a low certainty, to speed up this function
    """
    mask = s > c_thresh
    for i in range(len(s)):
        if mask[i] == False:
            continue
        for j in range(i + 1, len(s)):
            if mask[j] == False:
                continue
            u = iou(b[i], b[j])
            if u > iou_thresh:
                mask[j] = False
    return mask


def take_photo(shutter_pos):
    """
    this function takes a photo with the running camera app on an android smartphone
    returns: the captured image
    """
    lastfile = subprocess.check_output('adb shell ls /mnt/sdcard/DCIM/Camera | grep 2019'.split(' ')).decode().split('\n')[-2]

    # take photo
    subprocess.run(('adb shell input tap %d %d' % shutter_pos).split(' '))

    # wait until new photo was taken
    while True:
        o = subprocess.check_output('adb shell ls /mnt/sdcard/DCIM/Camera | grep 2019'.split(' ')).decode()
        f = o.split('\n')[-2]
        if f != lastfile:
            break

    # copy the image to the computer and remove the image from the smartphone
    subprocess.run(f"adb pull /mnt/sdcard/DCIM/Camera/{f} tmp/".split(' '))
    subprocess.run(f"adb shell rm /mnt/sdcard/DCIM/Camera/{f}".split(' '))
    return cv2.imread(f"tmp/{f}")


def make_message(m):
    """
    this is a helper function to create an image that shows a certain message
    m: a message
    returns: an image with the message on it
    """
    text_font = cv2.FONT_HERSHEY_DUPLEX
    text_size = 1
    text_width, text_height = cv2.getTextSize(m, text_font, text_size, 1)[0]
    x, y = (640 - text_width) / 2, (640 - text_height) / 2
    img = np.ones((640, 640, 3), dtype=np.uint8) * 255
    cv2.putText(img, m, (int(x), int(y)), text_font, text_size, (0,0,0))
    return img


if not os.path.exists('tmp'):
    os.mkdir('tmp')

window_name = 'prev'

# create window
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1800, 1800)
cv2.imshow(window_name, make_message('Press Space :)'))

# create messages that tell the user what is going on
taking_photo = make_message('Taking Photo...')
sending_file_to_server = make_message('Sending File to Server...')
predicting_food = make_message('Predicting Food...')

args = parser.parse_args()

shutter_pos = tuple(map(int, args.shutter_pos.split(',')))

# connect to the server
TCP_IP = args.ip
TCP_PORT = args.port
s = socket.socket()
s.connect((TCP_IP, TCP_PORT))

frame = None
while True:
    # listen for user input
    while True:
        k = cv2.waitKey(0)

        if k == ord('s') and frame is not None:
            # save last image to disk
            print(frame.shape)
            cv2.imwrite(f"{hash(str(frame))}.jpg", frame)
        if k == 32:
            # space was pressed. Take photo, make predictions and show image
            break
        if k == ord('q'):
            # quit
            s.close()
            exit()

    # take photo, and resize to 640x640
    cv2.imshow(window_name, taking_photo)
    cv2.waitKey(5)
    frame = take_photo(shutter_pos)
    frame = frame[:frame.shape[0], :frame.shape[0]]
    assert frame.shape[0] == frame.shape[1]
    frame = cv2.resize(frame, (640, 640))

    # encode image
    result, img = cv2.imencode('.jpg', frame[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])

    # send image to server and receive predictions
    cv2.imshow(window_name, sending_file_to_server)
    cv2.waitKey(5)
    send_np_array(s, img)
    cv2.imshow(window_name, predicting_food)
    cv2.waitKey(20)
    bb = receive_np_array(s, np.float32)
    classes = receive_np_array(s, np.uint8)
    scores = receive_np_array(s, np.float32)

    # show image and predictions
    frame = show_image(frame, bb, classes, scores)
