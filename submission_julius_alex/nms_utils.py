""" this implements the interclass non-maximum suppression"""


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


def non_maximum_supression(b, s, iou_thresh=.98, c_thresh=0.):
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