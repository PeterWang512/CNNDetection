import os
import sys
import torch
import torch.nn
import argparse
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from networks.resnet import resnet50


input_path = sys.argv[1]
model_path = sys.argv[2]

model = resnet50(num_classes=1)
state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(state_dict['model'])
model.cuda()
model.eval()

trans = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = trans(Image.open(input_path).convert('RGB'))

with torch.no_grad():
    in_tens = img.unsqueeze(0).cuda()
    prob = model(in_tens).sigmoid().item()

print('probability of being synthetic: {:.2f}%'.format(prob * 100))
