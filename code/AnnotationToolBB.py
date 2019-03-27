"""
With this messy script we labeled our data.
Use the mouse to draw bounding boxes.
Select the class with the buttons at the top.
"n" for next image, "p" for previous image
"u" to undo the last bounding box
"t" to toggle text rendering (useful if the tool lags)
"q" to quit
"""
import os, sys
from tkinter import *
from PIL import ImageTk, Image
import random
import argparse
import time
import numpy as np

parser = argparse.ArgumentParser(description='Annotate images')
parser.add_argument('--dir', type=str, required=True, help='Where images are located')
parser.add_argument('--correct', action='store_true', help='If just correcting labels')
parser.add_argument('--resize', type=float, default=1.0, help='Resize opened window')
parser.add_argument('--start_at', type=int, default=0, help='Start at specified item in list')
args = parser.parse_args()

category_names = ['Bier', 'Bier Mass', 'Weissbier', 'Cola', 'Wasser', 'Curry-Wurst', 'Weisswein',
                   'A-Schorle', 'Jaegermeister', 'Pommes', 'Burger', 'Williamsbirne', 'Alm-Breze', 'Brotzeitkorb',
                   'Kaesespaetzle']
count = args.start_at
timer = time.time()
track_time = []


def set_class(i):
    """ sets the current class """
    global cur_class
    buttons[cur_class]['relief'] = RAISED
    cur_class = i
    buttons[cur_class]['relief'] = SUNKEN


def next_image(previous):
    global cur_image, img, count, timer, track_time
    """ writes the current state to disk and loads the next image """
    if previous is not None:
        write()
        if previous:
            count -= 1
        else:
            count += 1
    cur_image = max(0, min(len(images_keys)-1, cur_image + (-1 if previous else 1))) if previous is not None else cur_image
    img = ImageTk.PhotoImage(Image.open(os.path.join(image_folder, images_keys[cur_image])).resize((int(1920 * resize), int(1080 * resize))))
    if images[images_keys[cur_image]] is None:
        images[images_keys[cur_image]] = []
    t_d = time.time() - timer
    track_time.append(t_d)
    items_left = len(images)-count
    median_estimate, mean_estimate = np.median(track_time)*items_left, np.mean(track_time)*items_left 
    timer = time.time()
    print("Labels: %d\t Images: %d/%d\tExpected remaining time (median/mean): %dmin / %dmin"%(sum(len(v) for v in images.values() if v is not None), count, len(images), median_estimate/60, mean_estimate/60))
    if args.correct and count >= len(images):
        destroy_window(None)


def mouse_click(e):
    """ if the mouse is clicked a bounding box is drawn"""
    bb = images[images_keys[cur_image]]
    if len(bb) > 0 and len(bb[-1]) == 3:
        bb[-1] += [e.x, e.y]
    else:
        bb.append([cur_class, e.x, e.y])


def mouse_move(e):
    global cvs
    mouse[:] = e.x, e.y


def undo(_):
    """ undo the last bounding box action"""
    bb = images[images_keys[cur_image]]
    if bb is not None and len(bb) > 0:
        if len(bb[-1]) == 3:
            del bb[-1]
        else:
            del bb[-1][3:]


def redraw():
    global img, mouse
    """ render the current image and the already drawn bounding boxes """
    cvs.delete('all')
    cvs.create_image(0, 0, anchor=NW, image=img)
    for x in images[images_keys[cur_image]]:
        c, x1, y1, x2, y2 = (x if len(x) == 5 else (x + mouse))
        cvs.create_rectangle(x1, y1, x2, y2, outline='#00FF00', width=2)
        if render_text:
            cvs.create_text(x1+2, y1+2, text=category_names[c], fill='#00FF00', anchor=NW)
    cvs.create_line(mouse[0], 0, mouse[0], img.height(), fill='#FF0000')
    cvs.create_line(0, mouse[1], img.width(), mouse[1], fill='#FF0000')
    root.after(50, redraw)


def write():
    """ writes all bounding boxes of all images to disk in OpenCV style i.e.
    "<filename> <number of labels> (<class> <x> <y> <width> <height> )*" """
    with open(to, 'w+') as f:
        for k in images_keys:
            bb = images[k]
            if bb is not None:
                f.write('%s %i %s\n' % (k, len(bb), ' '.join([('%d %d %d %d %d' % (l,
                int(min(x1, x2) / resize), int(min(y1, y2) / resize), int(abs(x2 - x1) / resize),
                int(abs(y2 - y1) / resize))) for l, x1, y1, x2, y2 in bb])))


def destroy_window(_):
    write()
    root.destroy()


def toggle_text():
    global render_text
    """ on some systems rendering text causes lags """
    render_text = not render_text


resize = args.resize
correcting = args.correct
random.seed(0)
image_folder = args.dir
to = os.path.join(image_folder, 'files.txt')
sum_labels = 0

# load the existing labels
if correcting and os.path.exists(to):
    images = {}
    with open(to, 'r') as f:
        for l in f.readlines():
            s = l.split(' ')
            images[s[0]] = [[int(s[2+5*i])] + [int(int(x) * resize) for x in s[3+5*i:7+5*i]] for i in range(int(s[1]))]
            for x in images[s[0]]:
                x[3] += x[1]
                x[4] += x[2]
            sum_labels += int(s[1])


else:
    images = {i: None for i in os.listdir(image_folder) if i[-3:] == 'jpg'}
    if os.path.exists(to):
        with open(to, 'r') as f:
            for l in f.readlines():
                s = l.split(' ')
                if s[0] in images.keys():
                    images[s[0]] = [[int(s[2+5*i])] + [int(int(x) * resize) for x in s[3+5*i:7+5*i]] for i in range(int(s[1]))]
                    for x in images[s[0]]:
                        x[3] += x[1]
                        x[4] += x[2]
                    sum_labels += int(s[1])
    else:
        print("No labels so far")

render_text = True
img = None
mouse = [0, 0]  # current mouse position
images_keys = list(images.keys())  # list of images
if not correcting:  # shuffle so that the images are not sorted by time
    random.shuffle(images_keys)
cur_class = 0
cur_image = args.start_at
for i in range(len(images_keys)):
    if images[images_keys[i]] is None:
        cur_image = i
        break

root = Tk()
cvs = Canvas(root, width=int(1920*resize), height=int(1080*resize))
cvs.bind('<Motion>', mouse_move)
cvs.bind('<Button-1>', mouse_click)
root.bind('q', destroy_window)  # quit
root.bind('n', lambda _: next_image(False))
root.bind('p', lambda _: next_image(True))
root.bind('u', undo)  # undo last bounding box
root.bind('t', lambda _: toggle_text())  # toggle text rendering if it causes lags
frame = Frame(root)

# add buttons to the window in order to change the class(category)
buttons = []
for i in range(len(category_names)):
    buttons.append(Button(frame, text=category_names[i], font=('Consolas', str(int(16*resize))),
                          command=(lambda x: lambda: set_class(x))(i)))
    buttons[-1].pack(side=LEFT, padx=5)
frame.pack()
cvs.pack()
set_class(category_names.index(image_folder) if image_folder in category_names else 0)
next_image(None)
redraw()
mainloop()

