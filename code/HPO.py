"""
Script that performs automated hyperparameter optimization on multiple gpus with given json config.
"""

import threading
import os
import sys
import json
import itertools
import random as rand
import shutil
import subprocess
from new_test import evaluate
from multiprocessing import Process, Manager
import numpy as np
import argparse as ap


parser = ap.ArgumentParser(description='Start hyperparameter search')
parser.add_argument('--config_file', type=str, required=True, help='Specify config file')
parser.add_argument('--train_dir', type=str, required=True, help='Specify train directory')
parser.add_argument('--gpus', type=str, default='01', help='Specify all gpus e.g. for 0 and 1 give --gpu 01')
parser.add_argument('--random', action='store_true', help='Do jobs in random order')
parser.add_argument('--sampling', type=int, default=0, help='Sample n random samples in search space')
parser.add_argument('--hyperparameter_config', type=str, default='', help='Give hyperparameter config as json file')


int_keys = ['num_steps', 'num_hard_examples', 'max_negatives_per_positive']


def dict_product(inp):
    return (dict(zip(inp.keys(), values)) for values in itertools.product(*inp.values()))


def random_sampling(inp, num_point=10):
    """
    Function to search randomly in search space.
    Input to be in form of list of two values (min, max)
    Returning random points in search space
    Only possible for numerical search options (no data augmentation)
    """
    low, high = [], []
    d = len(inp)
    for key in inp:
        l, h = inp[key][0], inp[key][-1]
        low.append(l)
        high.append(h)
    samples = np.random.uniform(low=low,high=high,size=(num_point,d))
    options = []
    for sample in samples:
        tmp_dict = {}
        for i, key in enumerate(inp):
            if key in int_keys:
                tmp_dict[key] = int(sample[i])
            else:
                tmp_dict[key] = sample[i]
        options.append(tmp_dict)
    return options


def modify_config(file_path, modify_name, modify_value, save_path=None):
    """
    modifies the tensorflow training config

    file_path: path to the config
    modify_name: property to modify
    modify_value: new value of the property
    save_path: where to save the new config. file_path if None
    """
    if save_path is None:
        save_path = file_path
    file = open(file_path, 'r')
    tmp_name = save_path+'_tmp'
    if os.path.exists(tmp_name):
        print("Error - TMP file already exists")
        return
    tmp_file = open(tmp_name, 'w+')
    for line in file.readlines():
        if modify_name in line:
            if modify_name[0] == '#':
                tmp_file.write(modify_value + '\n')
            else:
                split = line.split(':')
                new_line_value = split[0]+': '+str(modify_value)+'\n'
                tmp_file.write(new_line_value)
        else:
            tmp_file.write(line)
    os.rename(tmp_name, save_path)
    tmp_file.close()
    if modify_name == 'learning_rate_base':
        modify_config(file_path, 'warmup_learning_rate', modify_value, save_path)


class HPO:
    """
    This class coordinates the parallel Hyper Parameter Optimization
    """

    def __init__(self, base_config_path, hyper_parameter, train_dir, random_sampling_num=0):
        self.base_config_path = base_config_path
        self.hyper_parameter = hyper_parameter
        self.train_dir = train_dir
        self.random_sampling_num = random_sampling_num
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)
        self.ser_path = os.path.join(train_dir, 'HPO.json')
        if random_sampling_num > 0:
            self.samples = random_sampling(hyper_parameter, random_sampling_num)
            self.jobs = [Job(str(i), os.path.join(train_dir, str(i)), d) for i, d in enumerate(self.samples)]
            print(self.jobs)
        else:
            self.jobs = [Job(str(i), os.path.join(train_dir, str(i)), d) for i, d in enumerate(dict_product(hyper_parameter))]

    def run(self, threads_per_gpu, random=False, gpus=None, random_sample=False):
        gpus = gpus if gpus is not None else ['0', '1', '2', '3']
        jbs = rand.sample(self.jobs, len(self.jobs)) if random else self.jobs
        threads = {('%s#%d' % (g, j)): None for g in gpus for j in range(threads_per_gpu)}
        jbi = 0

        while jbi < len(jbs):

            for k in threads.keys():
                v = threads[k]
                if v is None or not v.is_alive():
                    while jbs[jbi].phase == 2:
                        jbi += 1
                    if not os.path.exists(jbs[jbi].path):
                        os.mkdir(jbs[jbi].path)
                    if not os.path.exists(jbs[jbi].config_path):
                        shutil.copyfile(self.base_config_path, jbs[jbi].config_path)
                    for k2, v2 in jbs[jbi].hp.items():
                        modify_config(jbs[jbi].config_path, k2, v2, jbs[jbi].config_path)

                    threads[k] = threading.Thread(target=jbs[jbi].run, args=(k.split('#')[0],))
                    threads[k].start()
                    jbi += 1

            print('Writing')
            with open(self.ser_path, 'w+') as f:
                json.dump(self.to_dict(), f)

            while True:
                for t in threads.values():
                    t.join(1)
                if any(not t.is_alive() for t in threads.values()):
                    break

        for t in threads.values():
            t.join()

    def to_dict(self):
        if self.random_sampling_num > 0:
            return {'base_config_path': self.base_config_path, 'hyper_parameter': self.samples,
                    'train_dir': self.train_dir, 'jobs': [j.to_dict() for j in self.jobs if j.phase > 0],
                    'random_sampling': self.random_sampling_num
                   }
        else: 
            return {'base_config_path': self.base_config_path, 'hyper_parameter': self.hyper_parameter,
                    'train_dir': self.train_dir, 'jobs': [j.to_dict() for j in self.jobs if j.phase > 0],
                    'random_sampling': self.random_sampling_num
                   }

    """
    Warning: Has to be extended to work with random sampling!
    """
    @staticmethod
    def from_json(path):
        with open(path, 'r') as f:
            s = json.load(f)
            hpo = HPO(s['base_config_path'], s['hyper_parameter'], s['train_dir'], random_sampling_num=s['random_sampling'])
            for j in s['jobs']:
                jj = hpo.jobs[int(j['iid'])]
                assert jj.iid == j['iid']
                jj.phase = j['phase']
        return hpo


class Job:
    """
    This class runs the training process and evaluates the model afterwards
    """
    def __init__(self, iid, path, hp, phase=0):
        self.iid = iid
        self.path = path
        self.config_path = os.path.join(self.path, 'pipeline.config')
        self.hp = hp
        self.phase = phase  # 0=not started, 1=started but not finished, 2=finished

    def run(self, gpu):
        print('Training %s on gpu %s' % (self.iid, gpu))
        self.phase = 1
        command = ['python', 'train.py', '--train_dir=%s' % self.path, '--pipeline_config_path=%s' % self.config_path, '--gpu=%s' % gpu]
        subprocess.run(command)        
        self.phase = 2
        command = ['python', 'export_inference_graph.py', '--input_type=image_tensor',
                   '--pipeline_config_path=%s' % self.config_path,
                   '--trained_checkpoint_prefix=%s' % os.path.join(self.path, 'model.ckpt-5000'),
                   '--output_directory=%s' % self.path, '--gpu=\'\'']
        subprocess.run(command)
        
        graph_path = os.path.join(self.path, 'frozen_inference_graph.pb')
        rpg, arpg = evaluate(graph_path, (300, 300), gpu, 90)

        print('Area under the curve: %.4f' % arpg)
        np.savez(os.path.join(self.path, 'evaluation'), rpg=rpg, arpg=arpg)

    def to_dict(self):
        return {'iid': self.iid, 'path': self.path, 'hp': self.hp, 'phase': self.phase}

    @staticmethod
    def from_dict(d):
        return Job(**d)

    
if __name__ == '__main__':
    arg = parser.parse_args()
    print("Parsed args: %s"%str(arg))
    """
    hp = {'#random_saturation': ['#random_saturation', 'data_augmentation_options {\n  random_adjust_saturation {\n    min_delta: 0.4\n    max_delta: 1.3\n  }\n }'],
          '#random_jitter_boxes': ['#random_jitter_boxes', 'data_augmentation_options {\n  random_jitter_boxes {\n    }\n }'],
          '#random_black_patches': ['#random_black_patches', 'data_augmentation_options {\n  random_black_patches {\n    max_black_patches: 15\n    probability: 0.666\n  }\n }'],
          '#%random_monochromatic_noise': ['#%random_monochromatic_noise', '#random_monochromatic_noise'],
          '#%random_rotation': ['#%random_rotation', '#random_rotation'],
          'learning_rate_base': ['0.005', '0.001']
          }
    """

    with open(arg.hyperparameter_config, 'r') as f:
        hp = json.load(f)
    
    
    json_path = os.path.join(arg.train_dir, 'HPO.json')
    if os.path.exists(json_path):
        hpo = HPO.from_json(json_path)
    else:
        hpo = HPO(arg.config_file, hp, arg.train_dir, random_sampling_num=arg.sampling)#HPO(bcp, hp, td)
    hpo.run(1, random=arg.random, gpus=list(arg.gpus))

