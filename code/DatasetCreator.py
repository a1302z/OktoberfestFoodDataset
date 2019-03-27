"""
This script
1. extracts relevant labels and videos from the raw data
2. splits the data into train, val and test
"""

import os
import pandas as pd
from dateutil import parser
from datetime import datetime
import numpy as np
import shutil
from sortedcontainers import SortedDict
import csv

# Path to the raw video data
data_path = '/nfs/students/winter-term-2018/project_2/video_data/videos/'
# Path to the raw labels
gtruth_path = '/nfs/students/winter-term-2018/project_2/labels/schanzerAlmData.csv'
camera = 'Cam1'
data_path = os.path.join(data_path, camera)

# dict that maps the indistinguishable category to the articleID in the raw labels
categories = {
    'Bier': [100, 102, 104],
    'Bier Maß': [101, 103, 105],
    'Weißbier': [106, 108, 110],
    'Cola': [113, 115, 116],
    'Wasser': [117, 121, 122, 128],
    'Curry-Wurst': [197, 198],
    'Weißwein': [129, 130, 131],
    'A-Schorle': [118],
    'Jägermeister': [173],
    'Pommes': [206],
    'Burger': [196],
    'Williamsbirne': [171],
    'Alm-Breze': [201],
    'Brotzeitkorb': [199],
    'Käsespätzle': [205]
}

category_ids = [x for v in categories.values() for x in v]

category_id_to_name = {i: n for n, v in categories.items() for i in v}

category_names = ['Bier', 'Bier Maß', 'Weißbier', 'Cola', 'Wasser', 'Curry-Wurst', 'Weißwein',
                   'A-Schorle', 'Jägermeister', 'Pommes', 'Burger', 'Williamsbirne', 'Alm-Breze', 'Brotzeitkorb',
                   'Käsespätzle']


def get_ground_truth(gt_path, lsv):
    """ This function extracts the relevant labels from the raw label file and returns them in a dict.
    A label is relevant if it contains an item from the selected indistinguishable categories
    and there exists a video clip at roughly the same time at which the order was placed.

    gt_path: path to the raw label file i.e. schanzerAlmData.csv
    lsv: dict of filenames of video clips as returned from ls_video_files

    returns: a dict,
             the keys are the timestamp of the corresponding video clip
             the values is the list of items visible on this video """

    gt = pd.read_csv(gt_path)
    gtr = SortedDict()
    for i, r in gt.iterrows():
        if r['intArticleID'] not in category_ids:  # if not one of the 15 selected indistinguishable categories
            continue

        # get the filename of the video clip to which this label belongs, if existent
        ts = parser.parse('%s %s' % (r['datSold'], r['timSold'])).timestamp()
        vts = get_video_timestamp(ts, lsv)

        if vts != -1:
            if vts not in gtr.keys():
                gtr[vts] = [0] * len(category_names)
            category_id = category_names.index(category_id_to_name[r['intArticleID']])
            gtr[vts][category_id] += int(r['floAmmount'])
    return gtr


def ls_video_files():
    """ This function returns the filenames of all video clips for the respective camera.

    returns: a dict where the key is the starting timestamp of the clip and the value is the filename """
    r = SortedDict()
    for i in range(17 if camera == 'Cam1' else 18, 28):  # the 17th is missing for Cam2, CamL and CamR
        p = os.path.join(data_path, '2018-05-%d' % i)
        r.update({int(f[:13]): f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f)) and f[0] != '.' and f[-3:] == 'mp4'})
    return r


def get_video_timestamp(ts, lsv):
    """
    ts: the timestamp of interest
    lsv: dict of filenames of video clips as returned from ls_video_files

    returns: the starting timestamp of the closest video clip before ts. If none exists in the past 18 seconds return -1 """
    index = lsv.bisect_right(int(ts) * 1000) - 1
    if not (0 <= index < len(lsv)):
        return -1
    t = lsv.keys()[index]
    return t if 0 <= (ts - t/1000) <= 18 else -1  # if more than 18s passed between the video clip and ts return -1


def sub_dict(d, ks):
    return {k: d[k] for k in ks}


def get_data_split(gt, ptrain, ptest):
    """ This function splits the video filenames (i.e. the dataset) into train, val and test

    gt: dict of the relevant labels as returned from get_ground_truth
    ptrain: train fraction
    ptest: test fraction

    returns: a dict with the split video filenames """
    s = list(gt.keys())
    np.random.shuffle(s)
    a, b = round(len(gt) * ptrain), round(len(gt) * ptest)  # pval = 1 - ptrain - ptest
    return {'train': sub_dict(gt, s[:a]), 'val': sub_dict(gt, s[a+b:]), 'test': sub_dict(gt, s[a:a+b])}


def copy_videos(gt, lsv, save_to):
    """ Copies all videos within gt to save_to

    gt: dict of the relevant labels
    lsv: dict of filenames of video clips as returned from ls_video_files
    save_to: path to where the videos should be copied """
    copy = set()
    for ts in gt.keys():
        copy.add((os.path.join(data_path, datetime.utcfromtimestamp(ts/1000).strftime('%Y-%m-%d'), lsv[ts]),
                  os.path.join(save_to, '%d.mp4' % ts)))
    for f, t in copy:
        shutil.copyfile(f, t)


def gt_to_numpy(gt):
    return np.array([[k] + v for k, v in gt.items()]).astype(np.int64)


def write_csv(d, path):
    with open(path, 'w') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        for r in d:
            wr.writerow(r)


def create_data_set(sp, lsv, save_to):
    """ Copies all videos and therefore creates the dataset

    sp: the split video file names as returned from get_data_split
    lsv: dict of filenames of video clips as returned from ls_video_files
    save_to: the path where the dataset should be created
    """
    if not os.path.exists(save_to):
        os.mkdir(save_to)
    for k, s in sp.items():  # copy train, val and test
        print('copying', k)
        p = os.path.join(save_to, k)
        if not os.path.exists(p):
            os.mkdir(p)

        gt_np = gt_to_numpy(s)
        gt_np = gt_np[gt_np[:, 0].argsort()]

        # save the labels as np array; and as csv for humans
        np.save(os.path.join(p, 'groundTruth'), gt_to_numpy(s))
        write_csv([['video_id', *category_names]] + list(gt_np), os.path.join(p, 'groundTruth.csv'))

        # copy all relevant videos
        copy_videos(s, lsv, p)


def print_split_statistics(sp):
    print(category_names)
    for k, s in sp.items():
        print(k)
        print(np.sum(list(s.values()), axis=0))


if __name__ == '__main__':
    np.random.seed(2)
    video_files = ls_video_files()  # get the filenames of the video clip
    gtruth = get_ground_truth(gtruth_path, video_files)  # get all relevant labels
    split = get_data_split(gtruth, .8, .1)  # split the dataset into train, val and test
    print_split_statistics(split)
    create_data_set(split, video_files, '/nfs/students/winter-term-2018/project_2/data_split')  # copy all relevant videos

