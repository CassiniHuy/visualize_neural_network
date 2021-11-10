import cv2
import os
from argparse import ArgumentParser
import numpy as np

argparser = ArgumentParser()
argparser.add_argument('--mask_path', type=str)
args = argparser.parse_args()

masks = os.listdir(args.mask_path)
print('masks:', len(masks))

m = []
for name in masks:
    mask_path = os.path.join(args.mask_path, name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) / 255
    mean = np.mean(mask.reshape([-1]))
    m.append(mean)
    print(name, mean)

print('mean:', sum(m) / len(m))
