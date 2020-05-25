
import argparse
import os
import csv
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score

from networks.resnet import resnet50

from IPython import embed

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d','--dir', type=str, default='examples_realfakedir')
parser.add_argument('-m','--model_path', type=str, default='weights/blur_jpg_prob0.5.pth')
parser.add_argument('-b','--batch_size', type=int, default=32)
parser.add_argument('-j','--workers', type=int, default=4, help='number of workers')
parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')

opt = parser.parse_args()

# Load model
model = resnet50(num_classes=1)
if(opt.model_path is not None):
    state_dict = torch.load(opt.model_path, map_location='cpu')
model.load_state_dict(state_dict['model'])
model.eval()
if(not opt.use_cpu):
    model.cuda()

# Transform
trans = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset loader
dataset = datasets.ImageFolder('example_realfakedir', transform=trans)
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=opt.batch_size,
                                          shuffle=False,
                                          num_workers=opt.workers)

y_true, y_pred = [], []
with torch.no_grad():
	for data, label in data_loader:
	    if(not opt.use_cpu):
	        data = data.cuda()
	    y_pred.extend(model(data).sigmoid().flatten().tolist())
	    y_true.extend(label.flatten().tolist())

y_true, y_pred = np.array(y_true), np.array(y_pred)
r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
acc = accuracy_score(y_true, y_pred > 0.5)
ap = average_precision_score(y_true, y_pred)

print('AP: {:2.2%}, Acc: {:2.2%}, Acc (real): {:2.2%}, Acc (fake): {:2.2%}'.format(ap, acc, r_acc, f_acc))


