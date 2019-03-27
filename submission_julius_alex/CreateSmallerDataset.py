"""
This script creates a dataset with fewer examples per class
"""

import os, sys
import numpy as np
import random
import shutil
random.seed(0)

# the size of the new dataset
size = 80

# path to the new and old dataset
path = 'thumbnails_sn_100'
path_new = 'thumbnails_sn_%d' % size

# first create a list of all files and their labels
files = []
with open(os.path.join(path, 'files.txt')) as f:
    for l in f.readlines():
        s = l.split(' ')
        classes = [0]*15
        for i in s[2::5]:
            classes[int(i)] += 1
        files.append((s[0], classes, l))


random.shuffle(files)
new_dataset = []
new_classdist = np.zeros(15)

# here the new dataset is created.
# the difficulty is that we can only add images,
# i.e. by adding an image we might add an example from an class that is underrepresented in the current dataset,
# but at the same time we might also add an example of a class that is overrepresented (which we do not want)
# therefore we use a greedy heuristic to choose which image we add next
# the heuristic maximizes the number of underrepresented classes while minimizing the number of overrepresented classes
while any(a < size for a in new_classdist) and len(files) > 0:
    avg = new_classdist.mean()
    best = None, -10000
    # find the image that maximizes the heuristic
    for i in range(len(files)):
        score = np.mean(((new_classdist <= avg) * 2 - 1) * np.minimum(files[i][1], 1))
        if score > best[1]:
            best = i, score
    # only add the image if it adds more underrepresented classes than overrepresented classes
    if best[1] >= 0:
        new_dataset.append(files[best[0]])
        new_classdist += new_dataset[-1][1]
    del files[best[0]]

# the number of examples per class might differ slightly, check manually if its ok
print(new_classdist)

# save the new dataset 
if len(sys.argv) > 1:
    if not os.path.exists(path_new):
        os.mkdir(path_new)
    with open(os.path.join(path_new, 'files.txt'), 'w+') as f:
        f.writelines(map(lambda x: x[2], new_dataset))
    for n, _, _ in new_dataset:
        shutil.copyfile(os.path.join(path, n), os.path.join(path_new, n))
else:
    print('not saving the new dataset')
