"""
Script to evaluate object detection model.
"""

import numpy as np
import sys, os
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import cv2
import argparse as ap
import copy
from multiprocessing import Process, Manager
from nms_utils import non_maximum_supression


parser = ap.ArgumentParser(description='Evaluate frozen graph')
parser.add_argument('--graph_path', dest='graph', type=str)
parser.add_argument('--gpu', type=str, default='3')
parser.add_argument('--reshape', type=str, default='300x300')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--per_class', action='store_true', default=False)
parser.add_argument('--merged', type=bool, default=False)
parser.add_argument('--normalize_saturation', action='store_true')
parser.add_argument('--data', type=str, default=None)
parser.add_argument('--ic_nms_thresh', type=float, default=1)


category_names = ['Bier', 'Bier Maß', 'Weißbier', 'Cola', 'Wasser', 'Curry-Wurst', 'Weißwein',
                   'A-Schorle', 'Jägermeister', 'Pommes', 'Burger', 'Williamsbirne', 'Alm-Breze', 'Brotzeitkorb',
                   'Käsespätzle']


def save_inference_for_batch(*args, **kwargs):
    """ Due to a bug in tensorflow the inference might crash.
    This function evaluates the function run_inference_for_batch until it succeeds """
    ret_dict = Manager().dict()
    def run():
        try:
            ret = run_inference_for_batch(*args, **kwargs)
            ret_dict['ret'] = ret
        except:
            print('Exception during evaluation, retrying')
    while not 'ret' in ret_dict:
        p = Process(target=run)
        p.start()
        p.join()
    return ret_dict['ret']


def run_inference_for_batch(images, graph, batch_size):
    """
    this function runs inference on the images i.e. predicts the bounding boxes, etc.
    images: numpy array of images
    graph: the graph into which the model was loaded
    batch_size: the whole test set does not fit into memory
    returns the bounding boxes, the detected classes and the confidence scores for the images
    """
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['detection_scores', 'detection_classes', 'detection_boxes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            ods = []
            for i in range(0, len(images), batch_size):
                ods.append(sess.run(tensor_dict, feed_dict={image_tensor: images[i:i+batch_size]}))
            output_dict = {k: np.concatenate([t[k] for t in ods]) for k in ods[0].keys()}
            output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int32) - 1
            return output_dict


def flatten_out_dict(output_dict, masks=None):
    """ This function removes the first dimension from the arrays returned from run_inference_for_batch
    if masks is not None it applies the masks (entries are removed because of e.g. non-maximum suppression) """
    if masks is None:
            return np.array(list(map(len, output_dict['detection_scores']))), output_dict['detection_scores'].flatten(), output_dict['detection_classes'].astype(np.int32).flatten()
    fmasks = masks.flatten()
    return np.array(list(map(sum, masks))), output_dict['detection_scores'].flatten()[fmasks], output_dict['detection_classes'].astype(np.int32).flatten()[fmasks]


def evaluate(path, resize, gpu, batch_size=100, data=None, eval_per_class = False, n_s_c = None, merged=False, confusion_matrix=False, normalize_saturation=False, threshold=0.2, ic_nms_thresh=1):
    """
    this method calculates the precision recall curve and the aug for the given model

    path: path to the frozen graph of the  model
    resize: size of the image that the model expects e.g. 300x300 or 640x640
    gpu: which gpu to use
    batch_size: the whole test set does not fit into memory e.g. for the SSD+FPN
    data: path to the test data created with CreateEvaluationData.ipynb
    eval_per_class: if the model should be evaluated per class
    n_s_c: data if the inference was already done
    merged: if Bier and Bier Maß was merged
    confusion_matrix: if the confusion matrix should be calculated
    normalize_saturation: if the saturation of the test images should be normalized
    threshold: the threshold for calculating the confusion matrix
    ic_nms_thresh: the iou threshold for non-maximum suppression. Set 1 to disable nms

    returns: the recall precision curve and the aug
    """
    # load the test images and the ground truth
    if data is None:
        data = np.load('/nfs/students/winter-term-2018/project_2/models/research/object_detection/training_folder_lsml/val_data.npz')
    elif type(data) is str:
        data = np.load(data)

    # limgs are the images i.e. this array is of shape (num_images, h, w, 3)
    # gt contains the ground truth i.e. is of shape (num_images, num_classes)
    # lmi is the x position of the left most bounding box. This is important to ensure that all objects of interest
    # are still on the image when the aspect ratio is changed (e.g. from 16:9 to 1:1)
    limgs, gt, lmi = data['imgs'], data['gt'], data['lmi']

    # if Bier and Bier Maß are merged
    if merged:
        gt[:, 0] += gt[:, 1]
        gt = np.delete(gt, np.s_[1], axis=1)

    if n_s_c is None:
        # if predictions need to be made
        # load the model into detection_graph
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # load and preprocess the test set images
        w, h = resize
        imgs = np.zeros((len(limgs), h, w, 3), dtype=np.float32)
        for i in range(len(imgs)):
            nw = int(limgs[i].shape[0] / h * w)
            m = min(limgs[i].shape[1] - nw, lmi[i])
            t = limgs[i]
            if normalize_saturation:
                t = cv2.cvtColor(t[:, :, ::-1], cv2.COLOR_BGR2HSV).astype(np.float64)
                t[:, :, 1] *= 114 / np.mean(t[:, :, 1])
                t = cv2.cvtColor(np.clip(t, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)[:, :, ::-1]
            t = t[:, m:m+nw, :]
            t = cv2.resize(t, (w, h))
            imgs[i] = t

        # run inference i.e. predict the bounding boxes, etc.
        od = save_inference_for_batch(imgs, detection_graph, batch_size)
        masks = None
        # apply interclass non-maximum suppression
        if ic_nms_thresh < 1:
            masks = np.array([non_maximum_supression(od['detection_boxes'][i], od['detection_scores'][i], iou_thresh=ic_nms_thresh) \
                 for i in range(len(od['detection_boxes']))])
        n, s, c = flatten_out_dict(od, masks)
    else:
        # if the predictions are already done
        n, s, c = n_s_c

    # n references the image i.e. the ith bounding box belongs to the n[i]th image
    # s is the confidence score
    # c is the class prediction

    n = np.repeat(np.arange(len(n)), n)
    # sort the predictions by their confidence score
    sort_indices = np.argsort(s)[::-1]
    n, c = n[sort_indices], c[sort_indices]

    # recall-precision curve
    rpg = np.empty((len(n)+2, 2))  # [recall, precision]

    tp, fp, fn = 0, 0, np.sum(gt)
    
    num_classes = len(category_names)
    cm = np.zeros((num_classes, num_classes))
    cm_gt = copy.deepcopy(gt)

    if not eval_per_class:
        # calculates the precision-recall curve
        for i in range(len(n)):
            if c[i] < num_classes and gt[n[i]][c[i]] > 0:
                # if detection is correct
                tp += 1
                fn -= 1
                gt[n[i], c[i]] -= 1
            else:
                # if detection is incorrect
                fp += 1
            rpg[i] = tp / (tp + fn), tp / (tp + fp)

        # ensure that the precision recall curve goes from 0 to 1
        rpg[-2:, :] = [rpg[-3, 0] + 1e-8, 0], [1., 0.]
        # smooth the precision recall curve
        rpg[:, 1] = [np.max(rpg[i:, 1]) for i in range(len(rpg))]
        # calculate the area under the curve
        arpg = np.trapz(y=rpg[:, 1], x=rpg[:, 0])
    else:
        # evaluation per class this is the same as above with
        # the only difference that detection of other classes are ignored
        rpg_list, arpg_list = [], []
        print("Evaluation for %d classes" % num_classes)
        for class_i in range(num_classes):
            rpg = [[0., 1.]]  # [recall, precision]
            tp, fp, fn = 0, 0, np.sum(gt[:, class_i])
            if fn == 0:
                print('Class %d, has no ground truth available!' % class_i)
            else:
                for i in range(len(n)):
                    if c[i] == class_i:
                        if gt[n[i]][c[i]] > 0:  # ground_truth of detected item in image i > 0
                            tp += 1
                            fn -= 1
                            gt[n[i], c[i]] -= 1
                        else:
                            fp += 1
                        rpg.append([tp / (tp + fn), tp / (tp + fp)])
            rpg += [[rpg[-1][0] + 1e-8, 0], [1., 0.]]
            rpg = np.array(rpg)
            rpg[:, 1] = [np.max(rpg[i:, 1]) for i in range(len(rpg))]
            arpg = np.trapz(y=rpg[:, 1], x=rpg[:, 0])
            rpg_list.append(rpg)
            arpg_list.append(arpg)
    
    if confusion_matrix:
        # Use all classifications where score is above threshold
        indices = s > threshold
        conf_img = n[indices]
        class_t = c[indices]

        # Creating a matrix that accumulates detections for each class in each picture (N x C)
        detection_matrix = np.zeros_like(cm_gt, dtype=np.float32)
        for i, img_num in enumerate(conf_img):
            detection_matrix[img_num, class_t[i]] += 1

        # Calculate difference between detections and groundtruth
        detection_difference = detection_matrix - cm_gt

        for h, line in enumerate(detection_difference):
            neg_ind = line < 0
            max_mis_detections = abs(np.sum(neg_ind*line))
            if max_mis_detections <= 0:
                continue

            for i, elm in enumerate(line):
                if detection_matrix[h, i] > 0 and cm_gt[h,i] > 0:
                    cm[i,i] += min(detection_matrix[h,i], cm_gt[h,i])
                if elm > 0:
                    for j, neg in enumerate(neg_ind):
                        if neg:
                            tmp = min(elm, abs(line[j]))
                            cm[j,i] += tmp
                
        # print the confusion matrix (Markup format)
        cm_str = ''
        cm_str += '| '
        for i in range(cm.shape[1]):
            cm_str += '| ' + category_names[i]
        cm_str += '| Class score |\n'
        for i in range(cm.shape[0]):
            cm_str += ' | ' + category_names[i]+' | '
            for j in range(cm.shape[1]):
                if cm[i,j] > 0.0:
                    cm_str += str(cm[i,j])
                cm_str += ' | '
            cm_str += '%.3f | '%arpg_list[i] 
            cm_str += '\n' 
        print(cm_str)
        
        print('\n%d results had higher score than threshold %.2f'%(conf_img.shape[0],threshold))
        print('Confusion matrix has %d entries'%np.sum(cm))

    # return the precision recall curve(s) and the aug(s)
    if eval_per_class:
        return rpg_list, arpg_list
    else:
        return rpg, arpg


if __name__ == '__main__':
    args = parser.parse_args()
    gpu = args.gpu
    batch_size = args.batch_size
    path_to_frozen_inference_graph = args.graph
    rescale = args.reshape
    data = args.data
    print("Given arguments: %s"%str(args))
    
    rpg, arpg = evaluate(path_to_frozen_inference_graph, list(map(int, rescale.split('x'))), gpu, batch_size, eval_per_class= args.per_class, merged=args.merged, normalize_saturation=args.normalize_saturation, data=data, ic_nms_thresh=args.ic_nms_thresh)
    
    if args.per_class:
        print('| Class | Score |')
        for i in range(len(arpg)):
            print("| %s \t| %.4f |"%(category_names[i], arpg[i]))
    else:
        print('Area under the curve: %.4f' % arpg)
        for r in np.arange(.05, 1, .05):
            c = rpg[rpg[:, 0] > r]
            p = c[0, 1] if len(c) > 0 else 0.
            print('precision@%.2f: %.4f' % (r, p))
