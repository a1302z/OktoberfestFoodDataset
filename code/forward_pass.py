"""
This file is for an automated forward pass to use in scripts.
"""


import argparse
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import os
import numpy as np
import cv2
from multiprocessing import Process, Manager
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime

category_names = ['Bier', 'Bier Mass', 'Weissbier', 'Cola', 'Wasser', 'Curry-Wurst', 'Weisswein',
                   'A-Schorle', 'Jaegermeister', 'Pommes', 'Burger', 'Williamsbirne', 'Alm-Breze', 'Brotzeitkorb',
                   'Kaesespaetzle']

"""
Was originally thought to be applicable to any given data. However it is more time consuming to bring data into npz format and than to use script that gives data as np array. Therefore this script is called from other scripts.

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=None, help='Specify model for forward pass')
parser.add_argument('--data', type=str, default=None, help='Specify data to passed through network')
parser.add_argument('--gpu', type=str, default='0', help='Which gpu to use')
parser.add_argument('--resize', type=str, default='640x640', help='Resize images to size')
parser.add_argument('--batch_size', type=int, default=2, help='Specify batch size')
parser.add_argument('--save_dir', type=str, default='results', help='Specify directory to save results')
"""

"""
Performs forward pass of certain model on given data.
If return_images is true it returns images with classification results into it. 
Otherwise it gives detection results as dictionary. 
"""
def forward_pass(model, data, gpu='0', resize=(640, 640), BATCH_SIZE=2, thresh=0.5, return_images=False):
    print("Start forward pass (%s)"%str(datetime.datetime.now()))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    lmi=np.zeros((len(data)))#*400
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    w, h = resize
    imgs = np.zeros((len(data), h, w, 3), dtype=np.float32)
    for i in range(len(imgs)):
        #nw = int(data[i].shape[0] / h * w)
        #m = min(data[i].shape[1] - nw, 0)
        t = data[i]#, :, m:m+nw, :]
        imgs[i] = cv2.resize(t, (w, h))
    t = time.time()
    out_dict = save_inference_for_batch(imgs, detection_graph, BATCH_SIZE)
    t = time.time() - t
    forward_time = float(t)/float(len(imgs))
    print("Whole inference took %s\nThis means %.4fs/image (%.3f fps)"%(seconds_to_str(t), forward_time, 1.0/forward_time))
    if not return_images:
        return out_dict
    n, s, c = flatten_out_dict(out_dict)
    n = np.repeat(np.arange(len(n)), n)
    sort_indices = np.argsort(s)[::-1]
    n, s, c = n[sort_indices], s[sort_indices], c[sort_indices]
    
    """
    Draw classification results into image data
    """
    oh, ow = data.shape[1], data.shape[2]
    t = time.time()
    for i in range(len(imgs)):
        bxs = out_dict['detection_boxes'][i]
        scr = out_dict['detection_scores'][i]
        clss = out_dict['detection_classes'][i]
        for j in range(len(bxs)):
            if scr[j] < thresh:
                continue
            x, y = int(ow*bxs[j,1]), int(oh*bxs[j,0])
            text = category_names[clss[j]]+ ': '+str(int(scr[j]*100))+'%'
            cv2.putText(data[i], text, (x, max(y-15, 0)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
            cv2.rectangle(data[i], (x,y), (int(ow*bxs[j,3]),int(oh*bxs[j,2])),(0,255,0),8)
    t = time.time() - t
    print("Image manipulation took %s"%seconds_to_str(t))
    print("Finished forward pass")
    return data
    

"""
Converts seconds to human readable time form.
"""    
def seconds_to_str(sec):
    sec_str = ''
    if sec > 3600:
        h = int(sec/3600)
        sec_str += '%dh '%h
        sec -= 3600*h
    if sec > 60:
        m = int(sec/60)
        sec_str += '%dm '%m
        sec -= 60*m
    sec_str += '%ds'%int(sec)
    return sec_str

    
def save_inference_for_batch(*args, **kwargs):
    #ret_dict = Manager().dict()
    success = False
    while not success:
        try:
            ret = run_inference_for_batch(*args, **kwargs)
            success = True
        except:
            print('Exception during evaluation, retrying')
    return ret 


def run_inference_for_batch(images, graph, batch_size):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['detection_scores','detection_classes', 'detection_boxes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            ods = []
            tx = time.time()
            for i in range(0, len(images), batch_size):
                ods.append(sess.run(tensor_dict, feed_dict={image_tensor: images[i:i+batch_size]}))
            tx = time.time() - tx
            print("Exact time for inference: %f s"%tx)
            output_dict = {k: np.concatenate([t[k] for t in ods]) for k in ods[0].keys()}
            output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int32) - 1
        sess.close()
        return output_dict

def flatten_out_dict(output_dict):
    return np.array(list(map(len, output_dict['detection_scores']))), output_dict['detection_scores'].flatten(), output_dict['detection_classes'].astype(np.int32).flatten()
    

