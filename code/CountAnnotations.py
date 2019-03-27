# coding: utf8
""" this script count the labels we did """

import os, sys
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--per_person', action='store_true', help='count labels per annotator?')
parser.add_argument('--dir', default=None, help='Specify directory')
args = parser.parse_args()

category_names = ['Bier', 'Bier Mass', 'Weissbier', 'Cola', 'Wasser', 'Curry-Wurst', 'Weisswein',
                   'A-Schorle', 'Jaegermeister', 'Pommes', 'Burger', 'Williamsbirne', 'Alm-Breze', 'Brotzeitkorb',
                   'Kaesespaetzle']

# who did which categories
categories_pp = {'Alex': [0, 3, 6, 9, 12], 'Vitalii':[1, 4, 7, 10, 13],'Julius':[2, 5, 8, 11, 14]}

# statistics for each person
labels_per_annotator = {'Alex':[0]*len(category_names),'Vitalii':[0]*len(category_names),'Julius':[0]*len(category_names)}
for a in categories_pp:
    print("%s assigned to %s"%(a, str([category_names[i] for i in categories_pp[a]])))

# number of labels per category
num = [0] * len(category_names)

# make sure that images that were labeled twice are only counted once
already_counted_images = set()

# count the annotations
for n in (os.listdir('.') if args.dir is None else [args.dir]):
    if os.path.exists(os.path.join(n, 'files.txt')):
        annotator = None
        for a in categories_pp:
            if n in [category_names[i] for i in categories_pp[a]]:
                annotator = a
        with open(os.path.join(n, 'files.txt')) as f:
            for l in f.readlines():
                s = l.split(' ')
                if s[0] in already_counted_images:
                    continue
                already_counted_images.add(s[0])

                for i in range(int(s[1])):
                    labels_per_annotator[annotator][int(s[2+5*i])] += 1
                    num[int(s[2 + 5 * i])] += 1
    elif os.path.isdir(n):
        print('No annotations for ' + n)

# print results
if args.per_person:
    line = '||'
    for a in labels_per_annotator:
        line += str(a)+'|'
    print(line)
    for i in range(len(labels_per_annotator[a])):
        line = '|'+category_names[i]+'|'
        for a in labels_per_annotator:
            if i in categories_pp[a]:
                line += '*%d*|' % labels_per_annotator[a][i]
            else:
                line += '%d|' % labels_per_annotator[a][i]
        print(line)
else:
    print("Total amount of labels per class:")
    for i in range(len(num)):
        print('|%s|%d|' % (category_names[i], num[i]))
print('Total number of labels', sum(num))
for a in labels_per_annotator:
    print("%s created overall %d labels"%(a,sum(labels_per_annotator[a])))

