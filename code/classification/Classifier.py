import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from AdditionalTransforms import *
from Data import ClassificationDataset
from Trainer import Trainer
import tensorflow as tf

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    RandomRotation90(),
    RandomSaturation(.4, 1.3),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.52476796, 0.5031609, 0.50243611], [0.09555659, 0.05939154, 0.07174332])
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.52476796, 0.5031609, 0.50243611], [0.09555659, 0.05939154, 0.07174332])
])

EPOCHS = 20
LR = 1e-3

train_dataset = ClassificationDataset('/nfs/students/winter-term-2018/project_2/data_split/train/experiments_julius', train_transforms)
val_dataset = ClassificationDataset('/nfs/students/winter-term-2018/project_2/data_split/val/experiments_julius', val_transforms)

result_path = '../../experiments_julius'
tensorboard_writer = tf.summary.FileWriter(result_path)
trainer = Trainer(train_dataset, val_dataset, tensorboard_writer=tensorboard_writer)

model = models.resnet18(pretrained=True)
optimizer = Adam(model.parameters(), lr=LR)
lr_scheduler = CosineAnnealingLR(optimizer, EPOCHS)
trainer.train(model, optimizer, lr_scheduler, 128, EPOCHS)
torch.save(model, result_path + '/model')
