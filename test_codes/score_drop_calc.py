import cv2
import os
import torch
from torchvision import models, transforms
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

argparser = ArgumentParser()
argparser.add_argument('--image_path', type=str)
argparser.add_argument('--mask_path', type=str)
args = argparser.parse_args()

totensor_and_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])
totensor = transforms.Compose([transforms.ToTensor()])

def load_input(img_path, mask_path=None):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224, 224))
    else:
        mask = np.zeros((224, 224), dtype='float32')
    blurred_img = np.float32(cv2.medianBlur(img, 11)) / 255
    img = np.float32(img) / 255
    mask = np.float32(mask) / 255

    img_tensor = totensor_and_normalize(img)[None, :,:,:]
    mask_tensor = totensor(mask)[None, :,:,:]
    blurred_img_tensor = totensor_and_normalize(blurred_img)[None, :,:,:]
    ipt = img_tensor.multiply(1 - mask_tensor) + blurred_img_tensor.multiply(mask_tensor)
    return ipt

def get_score(model, ipt, category=None):
    target = torch.nn.Softmax(dim=-1)(model(ipt)).cpu().detach().numpy()
    if category is None:
        category = np.argmax(target)
    return category, target[0, category]

imgs = os.listdir(args.image_path)
print('images:', len(imgs))

model = models.googlenet(pretrained=True)
model.eval()
model.cuda()

s = []
with torch.no_grad():
    for img in imgs:
        img_path = os.path.join(args.image_path, img)
        mask_path = os.path.join(args.mask_path, img)
        ipt = load_input(img_path).cuda()
        c, score0 = get_score(model, ipt)
        ipt = load_input(img_path, mask_path).cuda()
        _, score1 = get_score(model, ipt, c)
        delta = (score0 - score1) / score0
        s.append(delta)
        if delta < 0:
            print(img, delta, score0, score1)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            plt.imshow(cv2.medianBlur(img, 11))
            plt.show()
            input()

print('mean drop percent:', sum(s) / len(s))
print('length:', len(s))
s = list(filter(lambda x: x > 0, s))
print('length:', len(s))
print('mean drop percent:', sum(s) / len(s))
