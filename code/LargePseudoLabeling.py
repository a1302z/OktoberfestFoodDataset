"""
This script iterates over video data and classifies images with a given model. 
Afterwards it saves found labels as new dataset to create a pseudo label dataset (semi-supervised learning)
It is designed to such that it is easily extendable to create weak labels in the future. 
"""


from forward_pass import forward_pass, flatten_out_dict, seconds_to_str
import numpy as np
import sys
import cv2
from fnmatch import fnmatch
import os
import imageio
import shutil
from difflib import SequenceMatcher
import time
import tensorflow as tf
from object_detection.utils import dataset_util

#Suppress tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = tf.app.flags
parser.DEFINE_string('output_data_path', '/nfs/students/winter-term-2018/project_2/models/research/object_detection/training_folder_lsml/data/pseudo_labels', 'Data storage path')
parser.DEFINE_string('suffix', '', 'Suffix for data')
parser.DEFINE_string('gpu', '3', 'Which GPU to use? Note that it does not support multi-GPU use')
parser.DEFINE_integer('batch_size', 1, 'What batch size to use')
parser.DEFINE_float('threshold', 0.5, 'What threshold to user for accepting detection')
parser.DEFINE_integer('filter_value', 5, 'How many frames has a detection be consistent to be accepted')
parser.DEFINE_integer('saving_interval', 50, 'After what number of videos should data be written to disk?')
parser.DEFINE_integer('print_interval', 10, 'Defines after what number of videos status is printed')
parser.DEFINE_string('days', None, 'Specify which days should be used (Format: 18,19,20)')
parser.DEFINE_string('cams', 'Cam1', 'Specify which cams should be used (Format: Cam1,Cam2)')
parser.DEFINE_string('model', '/nfs/students/winter-term-2018/project_2/models/research/object_detection/training_folder_lsml/train/ssd_julius_mobilefpn_large/frozen_inference_graph.pb', 'Path to used model')
parser.DEFINE_boolean('limit_videos_per_day', False, 'If true only 10 first videos per day are considered')
parser.DEFINE_boolean('include_empty', False, 'If true include images that are classified as empty to final output')
parser.DEFINE_string('f', '', 'kernel')
flags = tf.app.flags.FLAGS
add_name = flags.suffix
"""
These do not need to be specified. Tensorflow requires them. 
"""
parser.DEFINE_string('output_path_train', flags.output_data_path+'_train_'+add_name, 'Path to output TFRecord train')
parser.DEFINE_string('output_path_eval', flags.output_data_path+'_eval_'+add_name, 'Path to output TFRecord eval')
parser.DEFINE_string('output_path', flags.output_data_path+add_name, 'Path to output TFRecord')
flags = tf.app.flags.FLAGS
#flags = parser.FLAGS



category_names = ['Bier', 'Bier Mass', 'Weissbier', 'Cola', 'Wasser', 'Curry-Wurst', 'Weisswein',
                   'A-Schorle', 'Jaegermeister', 'Pommes', 'Burger', 'Williamsbirne', 'Alm-Breze', 'Brotzeitkorb',
                   'Kaesespaetzle']

"""
Deletes all files in a given folder
"""
def empty_directory(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)

"""
Returns all file names in a given folder matching some pattern
"""
def get_files(path, pattern, not_pattern = None, printout=True):
    found = []
    for path, subdirs, files in os.walk(path):
        for name in files:
            if fnmatch(name, pattern) and (not_pattern is None or not fnmatch(name, not_pattern)):
                found.append(os.path.join(path, name))
    if printout:
        print("Found %d files in path %s"%(len(found), path))
    return found


"""
Bring labeled image into correct form for dataset
"""
def create_tf_example(img_data, out_dict, index, empty=False, thresh=0.1):
    example = img_data[0][index]
    example = cv2.cvtColor(example, cv2.COLOR_RGB2BGR)
    height = example.shape[0] # Image height
    width = example.shape[1] # Image width
    filename = str.encode('') # Filename of the image. Empty if image is not from file
    encoded_image_data = cv2.imencode('.png', example)
    encoded_image_data = encoded_image_data[1].tostring() # Encoded image bytes
    image_format = str.encode('png') # b'jpeg' or b'png'

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
             # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)
    
    if not empty:
        bxs = out_dict['detection_boxes'][index]
        clss = out_dict['detection_classes'][index]
        scr = out_dict['detection_scores'][index]
        for j in range(len(clss)):
            if scr[j] < thresh:
                continue
            classes.append(clss[j])
            classes_text.append(str.encode(category_names[clss[j]]))
            ymins.append(bxs[j][0])
            xmins.append(bxs[j][1])
            ymaxs.append(bxs[j][2])
            xmaxs.append(bxs[j][3])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

"""
Deletes all set tensorflow flags
"""
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)
        
        
"""
Filters given labels such that only labels that are consistent over n images are considered
"""
def filter_labels(out_dict, thresh=0.5, filter_value=5):
    """
    We filter noise by removing labels that are not consistent over at least n>1 images
    """
    previous_detection = None
    last_change = 0
    begin = 0
    already_printed = False
    filtered_image_assignments = [] #list of tuples with ([begin img, end img], dictionary)
    print_stuff = False
    for i in range(len(out_dict['detection_classes'])):
        bxs = out_dict['detection_boxes'][i]
        clss = out_dict['detection_classes'][i]
        scr = out_dict['detection_scores'][i]
        detection = None
        for j in range(len(clss)):
            if scr[j] < thresh:
                continue
            if detection is None:
                detection = {i: 0 for i in range(len(category_names))}
            detection[int(clss[j])] += 1
        """
        Handling detections that have been the same on the previous image. 
        Note that due to the code structure nothing but output operations needs to be performed here. 
        """
        if detection == previous_detection:
            if detection is not None and i - last_change > filter_value and not already_printed:
                out_str = "Detected items for image %i"%(i-filter_value)
                for key in detection:
                    if detection[key] > 0:
                        out_str += "\n  - %d %s"%(detection[key], category_names[key])
                if print_stuff:
                    print(out_str)
                already_printed = True
        
        else:
            """
            Handling changed detections. 
            If a detection changes and it was consistent for more than n images where n is the filter value, they get stored to the output.
            """
            if previous_detection is not None and i - last_change - 1 > filter_value:
                if print_stuff:
                    print("until image %d\n"%i)
                filtered_image_assignments.append(([begin, i], previous_detection))
            last_change = i
            already_printed = False
            begin = i+1
        previous_detection = detection
    if print_stuff:
        print("Found %d filtered assignments"%len(filtered_image_assignments))
    return filtered_image_assignments

"""
Converts bytes to human readable format
Source: https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
"""
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

"""
Labels and saves given image data to output dataset
"""
def save_to_disk(img_data, writer, gpu, batch_size, thresh, filter_value, include_empty=False):
    img_data = [np.vstack(img_data[i]) for i in range(len(img_data))]
    out_dict = forward_pass(model, img_data[0], gpu=gpu, BATCH_SIZE=batch_size, return_images=False)
    filtered_image_assignments = filter_labels(out_dict, thresh=thresh, filter_value=filter_value)
    last = 0
    save = False
    for filtered_label in filtered_image_assignments:
        """
        We thought that it might be interesting if the model can learn robustness from empty training images and therfore provided the option to include images that were classified as empty. 
        However, due to the large file sizes we reached we did not use this option. 
        """
        if include_empty:
            for i in range(last, filtered_label[0][0]):
                tf_example = create_tf_example(img_data, None, i, empty=True)
                writer.write(tf_example.SerializeToString())
        for i in range(filtered_label[0][0], filtered_label[0][1]):
            tf_example = create_tf_example(img_data, out_dict, i, empty=False, thresh=thresh)
            save = True
            writer.write(tf_example.SerializeToString())
        last = filtered_label[0][1]+1
    return save

def reset_img_data():
    global img_data, video_count
    video_count = 0
    img_data = [[] for c in cams]
    
def print_separator():
    print('\n---------------------------\n')
    
def args_to_list(arg, single_type=int):
    if arg is not None:
        if ',' in arg:
            x = arg.split(',')
        else:
            x = [single_type(arg)]
        return x
        
"""
Begin of actual Script
Settings from command line arguments
"""
        
days = range(18,28) if flags.days is None else args_to_list(flags.days, single_type=int)
cams = ['Cam1'] if flags.cams is None else args_to_list(flags.cams, single_type=str)

model = flags.model
print_every = flags.print_interval
"""
Videos are first limited to 10 videos for test mode. If flag limit_videos_per_day is given then it is kept otherwise all videos are used later on. 
"""
picked_videos = range(0,10)#500,50) #[0, -1]
all_images = not flags.limit_videos_per_day
videos_per_iteration=flags.saving_interval
GPU = flags.gpu
thresh = flags.threshold
filter_value = flags.filter_value
batch_size = flags.batch_size


## TfRecord settings
include_empty = flags.include_empty
out_file = flags.output_path
print("Writer will write to %s"%out_file)
if os.path.isfile(out_file):
    os.remove(out_file)
writer = tf.python_io.TFRecordWriter(out_file)




"""
End of Settings
"""


"""
Find videos corresponding to given settings
"""
videos_list = [[] for i in range(len(cams))]
for d in days:
    d_s = '2018-05-'+str(d)
    for i, cam in enumerate(cams):
        video_folder_id = cam+'/'+d_s
        video_path = '/nfs/students/winter-term-2018/project_2/video_data/videos/'+ video_folder_id
        pattern = "*.mp4"
        videos_list[i].append(get_files(video_path, pattern, not_pattern='*._*', printout=False))# for video_path in video_paths]
for i in range(len(cams)):
    print("Found %d days with %d total files for cam %d"%(len(videos_list[i]), sum([len(videos_list[i][j]) for j in range(len(videos_list[i]))]), i))
for i in range(len(videos_list)):
    for j in range(len(videos_list[i])):
        videos_list[i][j].sort()
        
img_data = [[] for c in cams]

video_count = 0
per_batch_time = []
batch_time = time.time()

"""
Main functionality of script is here. 
Iterate over each camera each day (each picked video). 
Load and extract images from videos, classify them and save create dataset by this.
"""
print("Start iterating over given videos")
for day in range(len(days)):
    """
    Change amount of videos used to all videos of that day. (Number dependent of day.)
    """
    if all_images:
        picked_videos = range(min([len(videos_list[i][day]) for i in range(len(cams))]))
    t_d = time.time()
    for c, i in enumerate(picked_videos):
        
        for j, cam in enumerate(videos_list):
            t_v = time.time()
            video = cam[day][i]
            images_video = []
            cap = cv2.VideoCapture(video)
            success,image = cap.read()
            count = 0
            while success and count < 90:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images_video.append(image)
                success,image = cap.read()
                count += 1
            try:
                img_data[j].append(np.array(images_video))
                if j == 0:
                    video_count += 1
            except Exception as e:
                """
                If saving interval is too large we get Memory issues. If these appear we get exceptions and half the saving interval.
                """
                print("Exception: %s"%str(e))
                print("Data size: %f"%sys.getsizeof(img_data[j]))
                videos_per_iteration /= 2
                print("REDUCING SAVING INTERVAL TO HALF! (n=%d)"%videos_per_iteration)
            
            
        """
        Every n videos save all labeled data to disk
        """
        if video_count >= videos_per_iteration:
            success = save_to_disk(img_data, writer, GPU, batch_size, thresh, filter_value, include_empty=include_empty)
            reset_img_data()
            per_batch_time.append(time.time() - batch_time)
            batch_time = time.time()
            med = np.median(per_batch_time)
            exp_time = ((len(picked_videos)-c-1)/videos_per_iteration)*med
            print_separator()
            if success:
                print("Successfully saved chunk to disk")
            print("Time per chunk: %s\nExpected time remaining (for day %d): %s"%(seconds_to_str(med), day, seconds_to_str(exp_time)))
            print_separator()
            
            
        if c % print_every == 0:
            t_v = time.time() - t_v
            print("%d/%d videos loaded (%.2fs per video)"%(c,len(picked_videos), t_v))
    t_d = time.time() - t_d
    print("Completed loading videos of day %d/%d\t Exp. time left: %s"%(day+1, len(days), seconds_to_str(t_d*(len(days)-day-1))))
if len(img_data[0]) > 0:
    save_to_disk(img_data, writer, GPU, batch_size, thresh, filter_value, include_empty=include_empty)
writer.close()
print_separator()
print("Created dataset has %s"%sizeof_fmt(os.path.getsize(out_file)))
print_separator()