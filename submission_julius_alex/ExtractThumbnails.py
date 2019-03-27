"""
This script extracts three frames per video clip from the dataset created by DatasetCreator
"""

import cv2
import numpy as np
import os

category_names = ['Bier', 'Bier Maß', 'Weißbier', 'Cola', 'Wasser', 'Curry-Wurst', 'Weißwein',
                   'A-Schorle', 'Jägermeister', 'Pommes', 'Burger', 'Williamsbirne', 'Alm-Breze', 'Brotzeitkorb',
                   'Käsespätzle']

# path to the split created by DatasetCreator.py
path = '/nfs/students/winter-term-2018/project_2/data_split/train/'  

# load the labels created by DatasetCreator.py
gt = np.load(os.path.join(path, 'groundTruth.npy'))  

# get the filenames of the videos that contain the specific category
categories = {}
for i in range(1, gt.shape[1]):
    categories[i-1] = gt[gt[:, i] != 0][:, 0]

print(len(categories))
for k, v in categories.items():
    folder = category_names[k] if k < len(category_names) else str(k)
    print(folder, len(v))
    # save the frames in a folder called thumbnails
    os.makedirs(os.path.join(path, 'thumbnails', folder), exist_ok=True)
    for t in v:
        vid = cv2.VideoCapture(os.path.join(path, '%d.mp4' % t))
        # extract frames number 20, 45 and 70
        for o in [20, 45, 70]:
            vid.set(cv2.CAP_PROP_POS_FRAMES, o)
            # save the frame as "<video_starting_timestamp>_<frame_number>.jpg"
            cv2.imwrite(os.path.join(path, 'thumbnails', folder, '%d_%d.jpg' % (t, o)), vid.read()[1])
        vid.release()

