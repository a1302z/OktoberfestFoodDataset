"""
This is the server for the live demo. It receives the image, predicts the bounding boxes and sends the predictions back.
"""
import numpy as np
import os, socket
import tensorflow as tf
import cv2
from Utils import receive_np_array, send_np_array
import argparse as ap


parser = ap.ArgumentParser(description='Live-Demo Server')
parser.add_argument('--ip', type=str, default='localhost')
parser.add_argument('--port', type=int, default=14441)
parser.add_argument('--model_path', type=str, required=True, help='The path to the frozen inference graph.')
parser.add_argument('--gpu', type=str, default='0')

class Predictor:
    """
    This class loads and runs the tensorflow model
    """
    def __init__(self, frozen_graph, gpu=None):
        """
        frozen_graph: the path to the model
        gpu: which gpu to use
        """
        # load model into detection_graph
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.graph = detection_graph
        with self.graph.as_default():
            self.sess = tf.Session()
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in ['detection_scores', 'detection_classes', 'detection_boxes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    def predict(self, image):
        """
        image: one image resized to the correct size (e.g. 640x640)
        returns: the predicted bounding boxes, classes, and confidence scores as a dict
        """
        with self.graph.as_default():
            output_dict = self.sess.run(self.tensor_dict, feed_dict={self.image_tensor: image[np.newaxis]})
            for k in output_dict.keys():
                output_dict[k] = output_dict[k][0]
            output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int32) - 1
            return output_dict


args = parser.parse_args()

# the ip and the port to open where the tcp client should be opened
TCP_IP = args.ip
TCP_PORT = args.port

# '/home/hansjako/ssd_julius_mobilefpn_large/frozen_inference_graph.pb'
p = Predictor(args.model_path, args.gpu)

# open the tcp connection and listen
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen(True)
s, _ = s.accept()

while True:
    try:
        # this could be made faster by making it async, i.e. that predicting, receiving and sending all work simultaneous. However, that is much more complicated.

        # receive the image from the client, already resized, decode it and predict the bounding boxes, etc.
        img = receive_np_array(s, np.uint8)
        img = cv2.imdecode(img, 1)
        out_dict = p.predict(img)

        # send the bounding boxes, the classes and the confidence scores back to the client
        send_np_array(s, out_dict['detection_boxes'].astype(np.float32))
        send_np_array(s, out_dict['detection_classes'].astype(np.uint8))
        send_np_array(s, out_dict['detection_scores'].astype(np.float32))

    except AttributeError:
        # if the tpc connection was closed by the client listen for new connection
        s.close()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(True)
        s, _ = s.accept()
    except KeyboardInterrupt:
        # if the script is killed close the socket
        s.close()
        break
