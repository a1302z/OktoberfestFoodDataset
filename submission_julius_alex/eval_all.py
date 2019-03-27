"""
As we have a large set of trained models we wanted to create a method to automatically find the best and store results.
This is an extension to new_test as it applies this script to all models it finds in given folders.
"""

import os,sys
sys.path.append("/nfs/students/winter-term-2018/project_2/models/research/")

import new_test as nt

from fnmatch import fnmatch
import argparse as ap
import json
import time, datetime
import numpy as np



parser = ap.ArgumentParser(description='Evaluate all frozen graphs in directory')
parser.add_argument('--dir', type=str, default='./', help='Specify root directory')
parser.add_argument('--gpu', type=str, default='0', help='GPU used for evaluation')
parser.add_argument('--save_file', type=str, default='results', help='Json file to store results')
args = parser.parse_args()

root = args.dir
pattern = "*.pb"
wrong = "saved_model.pb"

graphs = []
"""
Find models that have .pb as ending but are not called saved_model.pb with regexes.
"""
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern) and not fnmatch(name, wrong):
            graphs.append(os.path.join(path, name))
print("Found %d files"%len(graphs))

best = None
best_score = 0.0
save_dict = {}
t_point = time.time()
t = []
sep = '\n---------------------------------------\n'
for i, g in enumerate(graphs):
    remaining = len(graphs) - i
    """
    Calculate estimate on remaining time
    """
    if len(t) > 0:
        rt_median, rt_mean = datetime.timedelta(seconds=remaining*np.median(t)), datetime.timedelta(seconds=remaining*np.mean(t))
    else: 
        rt_median, rt_mean = 'Not enough data', 'Not enough data'
    print(sep+("Evaluation of %s (%d/%d)\nExpected remaining time (median/mean): %s/%s"+sep)%(g, i, len(graphs), str(rt_median), str(rt_mean)))
    try:
        rpg, arpg = nt.evaluate(g, resize=(300,300), gpu=args.gpu, batch_size = 10)
        t.append(time.time()- t_point)
        t_point = time.time()
        g_dict = {}
        g_dict['evaluation_date'] = str(datetime.datetime.now())
        g_dict['score'] = arpg
        save_dict[g] = g_dict
        if arpg > best_score:
            best = g
            best_score = arpg
    except Exception as e: 
        print("Could not evaluate %s"%g)
        print("Exception: %s"%str(e))

"""
Save results to json file
"""
with open(os.path.join(args.dir, args.save_file)+'.json', 'w+') as f:
    json.dump(save_dict, f)        
print("\n")
if best is None:
    print("Did not found graph with score > 0.0")
else:
    print("Best score %.4f was achieved by %s"%(best_score, best))