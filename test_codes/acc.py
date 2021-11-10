import cv2
import os
import torch
from torchvision import models, transforms
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

argparser = ArgumentParser()
argparser.add_argument('--image_path', type=str)
argparser.add_argument('--rand_layer', type=str)
argparser.add_argument('--label', type=int)
args = argparser.parse_args()

totensor_and_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])
totensor = transforms.Compose([transforms.ToTensor()])

def load_input(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img_tensor = totensor_and_normalize(img)[None, :,:,:]
    return img_tensor

def get_label(model, ipt):
    target = torch.nn.Softmax(dim=-1)(model(ipt)).cpu().detach().numpy()
    category = np.argmax(target)
    return category

imgs = os.listdir(args.image_path)
print('images:', len(imgs))

model = models.googlenet(pretrained=True)

found = False
for name, layer in model._modules.items():
    if layer is None:
        continue
    if name == args.rand_layer:
        found = True
    if found is True:
        for param in layer.parameters():
            param.data = torch.randn_like(param)

model.eval()
model.cuda()

s = []
with torch.no_grad():
    for img in imgs:
        img_path = os.path.join(args.image_path, img)
        ipt = load_input(img_path).cuda()
        output = get_label(model, ipt)
        s.append(output == args.label)

print('accuracy:', sum(s) / len(s))
