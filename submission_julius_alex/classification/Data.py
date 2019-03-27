import torch
import os
import numpy as np
from os.path import join, exists
from skimage import io, transform
from torch.utils.data import Dataset


def rectangle_overlap(ra, rb):
    return min(ra[0] + ra[2], rb[0] + rb[2]) - max(ra[0], rb[0]) > 0 and \
           min(ra[1] + ra[3], rb[1] + rb[3]) - max(ra[1], rb[1]) > 0


def extract_images(img, cbb, width=224, height=224):
    for i in range(len(cbb)):
        while True:
            xn, yn = np.random.randint(0, img.shape[1] - width - 1), np.random.randint(0, img.shape[0] - height - 1)
            if not any(rectangle_overlap(cbb[j][1:], (xn, yn, width, height)) for j in range(len(cbb))):
                break
        r = np.copy(img[yn:yn+height, xn:xn+width])
        x, y, w, h = cbb[i][1:]
        r2 = img[y:y+h, x:x+w]
        if w > width or h > height:
            if w > h:
                scale = width/w
            else:
                scale = height/h
            r2 = np.clip(transform.resize(r2, (int(h*scale), int(w*scale))) * 255, 0, 255).astype(np.uint8)
        a, b = np.random.randint(0, width - r2.shape[1] + 1), np.random.randint(0, height - r2.shape[0] + 1)
        r[b:b+r2.shape[0], a:a+r2.shape[1]] = r2
        yield cbb[i][0], r


def create_dataset(path, out):
    if not exists(out):
        os.mkdir(out)
    for i in range(15):
        t = join(out, str(i))
        if not exists(t):
            os.mkdir(t)
    for x in os.listdir(path):
        files = os.path.join(path, x, 'files.txt')
        if not os.path.exists(files):
            continue
        with open(files, 'r') as f:
            for l in map(lambda z: z.split(' '), f.readlines()):
                if l[1] == '0':
                    continue
                img = io.imread(os.path.join(path, x, l[0]))
                cbb = [tuple(map(int, l[2+i*5:7+i*5])) for i in range(int(l[1]))]
                for e, (c, i) in enumerate(extract_images(img, cbb)):
                    io.imsave(join(out, str(c), f'{l[0].split(".")[0]}_{e}.png'), i)


class ClassificationDataset(Dataset):

    def __init__(self, path, transforms):
        self.data = []
        for i in range(15):
            for f in os.listdir(join(path, str(i))):
                self.data.append((join(path, str(i), f), i))
        self.transforms = transforms

    def __getitem__(self, i):
        p, c = self.data[i]
        img = io.imread(p)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, torch.tensor(c)

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    create_dataset('/nfs/students/winter-term-2018/project_2/data_split/val/',
                   '/nfs/students/winter-term-2018/project_2/data_split/val/experiments_julius')
    create_dataset('/nfs/students/winter-term-2018/project_2/data_split/train/thumbnails-saturation-normalized/',
                   '/nfs/students/winter-term-2018/project_2/data_split/train/experiments_julius')
