import configparser
import argparse
import os
import torch
from utils import get_config
from mask import Mask
import cv2

argparser = argparse.ArgumentParser()
argparser.add_argument('--image_path', type=str)
argparser.add_argument('--mask_path', type=str)
argparser.add_argument('--rand_layer', type=str)
args = argparser.parse_args()

cfg = configparser.ConfigParser()
cfg.read('config.ini')


cfg = get_config('config.ini').MASK

from model_wrapper import get_model_wrapper

model = get_model_wrapper('GoogleNetWrapper')()
found = False
for name, layer in model.model._modules.items():
    if layer is None:
        continue
    if name == args.rand_layer:
        found = True
    if found is True:
        for param in layer.parameters():
            param.data = torch.randn_like(param)

model.model.cuda()
mask_engine = Mask(model, cfg)
_, _, mask, cam = mask_engine(args.image_path)
cv2.imwrite(os.path.join(args.mask_path, args.rand_layer + '.jpg'), mask)
