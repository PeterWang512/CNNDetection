
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

from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d','--dir', nargs='+', type=str, default='examples/realfakedir')
parser.add_argument('-m','--model_path', type=str, default='weights/blur_jpg_prob0.5.pth')
parser.add_argument('-b','--batch_size', type=int, default=32)
parser.add_argument('-j','--workers', type=int, default=4, help='number of workers')
parser.add_argument('-c','--crop', type=int, default=None, help='by default, do not crop. specify crop size')
parser.add_argument('--use_cpu', action='store_true', help='uses gpu by default, turn on to use cpu')
parser.add_argument('--size_only', action='store_true', help='only look at sizes of images in dataset')

opt = parser.parse_args()

# Load model
if(not opt.size_only):
  model = resnet50(num_classes=1)
  if(opt.model_path is not None):
      state_dict = torch.load(opt.model_path, map_location='cpu')
  model.load_state_dict(state_dict['model'])
  model.eval()
  if(not opt.use_cpu):
      model.cuda()

# Transform
trans_init = []
if(opt.crop is not None):
  trans_init = [transforms.CenterCrop(opt.crop),]
  print('Cropping to [%i]'%opt.crop)
else:
  print('Not cropping')
trans = transforms.Compose(trans_init + [
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset loader
if(type(opt.dir)==str):
  opt.dir = [opt.dir,]

print('Loading [%i] datasets'%len(opt.dir))
data_loaders = []
for dir in opt.dir:
  dataset = datasets.ImageFolder(dir, transform=trans)
  data_loaders+=[torch.utils.data.DataLoader(dataset,
                                          batch_size=opt.batch_size,
                                          shuffle=False,
                                          num_workers=opt.workers),]

y_true, y_pred = [], []
Hs, Ws = [], []
with torch.no_grad():
  for data_loader in data_loaders:
    for data, label in tqdm(data_loader):
    # for data, label in data_loader:
      Hs.append(data.shape[2])
      Ws.append(data.shape[3])

      y_true.extend(label.flatten().tolist())
      if(not opt.size_only):
        if(not opt.use_cpu):
            data = data.cuda()
        y_pred.extend(model(data).sigmoid().flatten().tolist())

Hs, Ws = np.array(Hs), np.array(Ws)
y_true, y_pred = np.array(y_true), np.array(y_pred)

print('Average sizes: [{:2.2f}+/-{:2.2f}] x [{:2.2f}+/-{:2.2f}] = [{:2.2f}+/-{:2.2f} Mpix]'.format(np.mean(Hs), np.std(Hs), np.mean(Ws), np.std(Ws), np.mean(Hs*Ws)/1e6, np.std(Hs*Ws)/1e6))
print('Num reals: {}, Num fakes: {}'.format(np.sum(1-y_true), np.sum(y_true)))

if(not opt.size_only):
  r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
  f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
  acc = accuracy_score(y_true, y_pred > 0.5)
  ap = average_precision_score(y_true, y_pred)

  print('AP: {:2.2f}, Acc: {:2.2f}, Acc (real): {:2.2f}, Acc (fake): {:2.2f}'.format(ap*100., acc*100., r_acc*100., f_acc*100.))


