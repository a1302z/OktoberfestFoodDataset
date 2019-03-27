import torch
from AdditionalTransforms import *
from Data import ClassificationDataset
import numpy as np
from sklearn.metrics import confusion_matrix

category_names = ['Bier', 'Bier Maß', 'Weißbier', 'Cola', 'Wasser', 'Curry-Wurst', 'Weißwein',
                   'A-Schorle', 'Jägermeister', 'Pommes', 'Burger', 'Williamsbirne', 'Alm-Breze', 'Brotzeitkorb',
                   'Käsespätzle']

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.52476796, 0.5031609, 0.50243611], [0.09555659, 0.05939154, 0.07174332])
])

val_dataset = ClassificationDataset('/nfs/students/winter-term-2018/project_2/data_split/val/experiments_julius', val_transforms)

model = torch.load('../../experiments_julius/model')
model.cuda()
y_true = []
y_pred = []
for i, l in val_dataset:
    y_true.append(l.numpy())
    y_pred.append(np.argmax(model(i[None].cuda()).detach().cpu().numpy()))
    #print(f'{y_pred[-1]} {y_true[-1]}')


print(confusion_matrix(y_true, y_pred))

