import configparser
import argparse
import os
from utils import get_config
from mask import Mask
import cv2

argparser = argparse.ArgumentParser()
argparser.add_argument('--image_path', type=str)
argparser.add_argument('--mask_path', type=str)
argparser.add_argument('--cam_path', type=str)
args = argparser.parse_args()

cfg = configparser.ConfigParser()
cfg.read('config.ini')


cfg = get_config('config.ini').MASK

from model_wrapper import get_model_wrapper

images = os.listdir(args.image_path)
print(len(images), 'images exists.')
masks = os.listdir(args.mask_path)
print(len(masks), 'masks existed already.')
images_undo = set(images) - (set(images) & set(masks))
print(len(images_undo), 'to be processing...')

n_done = len(images_undo)
i = 0
for image in images_undo:
    model = get_model_wrapper('GoogleNetWrapper')()
    model.model.cuda()
    mask_engine = Mask(model, cfg)
    img_path = os.path.join(args.image_path, image)
    _, _, mask, cam = mask_engine(img_path)
    cv2.imwrite(os.path.join(args.mask_path, image), mask)
    cv2.imwrite(os.path.join(args.cam_path, image), cam)
    i += 1
    print(str(i) + '/' + str(n_done), 'done,', 'processing: ', image)
