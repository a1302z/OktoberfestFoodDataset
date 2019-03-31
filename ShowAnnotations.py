"""
This script shows the labels created with the annotation tool.
example usage: python ShowAnnotations.py Bier
"q" to quit
"n" for the next image
"p" for the previous image
"""

import cv2
import sys, os

category_names = ['Bier', 'Bier Mass', 'Weissbier', 'Cola', 'Wasser', 'Curry-Wurst', 'Weisswein',
                   'A-Schorle', 'Jaegermeister', 'Pommes', 'Burger', 'Williamsbirne', 'Alm-Breze', 'Brotzeitkorb',
                   'Kaesespaetzle']

if len(sys.argv) < 2:
    print("Please specify a folder!")
    exit()
path = sys.argv[1]
resize = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
print("Resizing display by a factor of %.2f" % resize)

# read the labels
with open(os.path.join(path, 'files.txt')) as f:
    ll = f.readlines()

j=0
while j < len(ll):
    # show images with labels
    l = ll[j]
    s = l.split(' ')
    img = cv2.imread(os.path.join(path, s[0]))
    for i in range(int(s[1])):
        c, x, y, w, h = [int(x) for x in s[2+5*i:7+5*i]]
        cv2.putText(img, category_names[c], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    img = cv2.resize(img, (int(img.shape[1]*resize), int(img.shape[0]*resize)))
    cv2.imshow('xxx', img)
    k = -1
    while k == -1:
        k = cv2.waitKey(20)
    if k == ord('q'):
        # quit
        break
    elif k == ord('p'):
        # previous image
        j -= 1
    elif k == ord('n'):
        # next image
        j += 1
    print("Displaying image %d/%d"%(j, len(ll)))
