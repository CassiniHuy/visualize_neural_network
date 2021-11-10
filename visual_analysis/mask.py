import torch
from torch.autograd import Variable
from torchvision import models
import torchvision.transforms as transforms
import cv2
import sys
import os
import numpy as np

class Mask():
    def __init__(self, model, cfg):
        ''' Read https://arxiv.org/abs/1704.03296 for understanding parameters '''
        self.model = model
        self.use_cuda = next(self.model.parameters()).is_cuda
        self.max_iterations = cfg.max_iterations
        self.learning_rate = cfg.learning_rate
        self.beta = cfg.beta
        self.lambda1 = cfg.lambda1
        self.lambda2 = cfg.lambda2
        self.tau = cfg.tau
        self.img_size = cfg.img_size
        self.mask_size = cfg.mask_size
        
        self.totensor_and_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
        self.totensor = transforms.Compose([transforms.ToTensor()])
        self.upsample = torch.nn.UpsamplingBilinear2d(size=self.img_size)
        if self.use_cuda:
            self.upsample = self.upsample.cuda()
    
    def tv_norm(self, input):
        img = input[0, 0, :]
        row_grad = torch.mean(torch.abs((img[:-1 , :] - img[1 :, :])).pow(self.beta))
        col_grad = torch.mean(torch.abs((img[: , :-1] - img[: , 1 :])).pow(self.beta))
        return row_grad + col_grad

    def l1_norm(self, mask):
        return torch.mean(torch.abs(1 - mask))
    
    def gen_noise(self, img):
        noise = torch.randn_like(img) * self.tau
        if self.use_cuda:
            noise = noise.cuda()
        return noise
    
    def loss(self, mask, img, blurred_img, category):
        upsampled_mask = self.upsample(mask)
        # The single channel mask is used with an RGB image, 
        # so the mask is duplicated to have 3 channel,
        upsampled_mask = upsampled_mask.expand(1, 3, upsampled_mask.size(2), upsampled_mask.size(3))
        # Use the mask to perturbated the input image.
        # # Generate noise and sample input
        perturbated_input = img + self.gen_noise(img)
        perturbated_input = perturbated_input.mul(upsampled_mask) + blurred_img.mul(1 - upsampled_mask)
        # Compute score
        outputs = torch.nn.Softmax(dim=-1)(self.model(perturbated_input))
        # Compute loss
        loss = self.lambda1 * self.l1_norm(mask) + self.lambda2 * self.tv_norm(mask) + outputs[0, category]
        return loss
    
    def get_cam(self, mask, img, blurred):
        mask = self.upsample(mask)
        mask = mask.cpu().data.numpy()[0]
        mask = np.transpose(mask, (1, 2, 0))

        mask = (mask - np.min(mask)) / np.max(mask)
        mask = 1 - mask
        heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
        
        heatmap = np.float32(heatmap) / 255
        cam = 1.0*heatmap + np.float32(img)/255
        cam = cam / np.max(cam)

        img = np.float32(img) / 255
        perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)    

        perturbated = np.uint8(255*perturbated)
        heatmap = np.uint8(255*heatmap)
        mask = np.uint8(255*mask)
        cam = np.uint8(255*cam)
        return perturbated, heatmap, mask, cam

    def __call__(self, img, category=None, generator=False):
        # Load images
        if isinstance(img, str):
            original_img = cv2.imread(img, cv2.IMREAD_COLOR)
            original_img = cv2.resize(original_img, self.img_size)
        else:
            original_img = img
        img = np.float32(original_img) / 255
        blurred_img1 = cv2.GaussianBlur(img, (11, 11), 5)
        blurred_img2 = np.float32(cv2.medianBlur(original_img, 11)) / 255
        blurred_img_numpy = (blurred_img1 + blurred_img2) / 2
        # Convert image to torch variables
        img = self.totensor_and_normalize(img)[None, :,:,:]
        blurred_img = self.totensor_and_normalize(blurred_img2)[None, :,:,:]
        # Get mask
        if category is None:
            with torch.no_grad():
                target = torch.nn.Softmax(dim=-1)(self.model(img.cuda() if self.use_cuda else img))
                category = np.argmax(target.cpu().data.numpy())

        mask_init = np.ones(self.mask_size, dtype=np.float32)
        mask = self.totensor(mask_init)[None, :,:,:]
        # To cuda
        if self.use_cuda:
            img, blurred_img, mask = img.cuda(), blurred_img.cuda(), mask.cuda()
        # Turn off model parameters grad record
        for p in self.model.parameters():
            p.requires_grad = False
        # Iterate
        mask.requires_grad_(True)
        optimizer = torch.optim.Adam([mask], lr=self.learning_rate)
        self.model.set_constraint(img)
        for i in range(self.max_iterations):
            if generator:
                yield self.get_cam(mask, original_img, blurred_img_numpy)
            loss = self.loss(mask, img, blurred_img, category)
            # if i % 10 == 0:
            #     print(i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mask.data.clamp_(0, 1)

        if generator:
            yield self.get_cam(mask, original_img, blurred_img_numpy)
        else:
            return self.get_cam(mask, original_img, blurred_img_numpy)

if __name__ == '__main__':
    import configparser
    import argparse
    from utils import get_config

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--image_path', type=str)
    argparser.add_argument('--config', type=str, default='config.ini')
    argparser.add_argument('--save_path', type=str, default='./examples')
    args = argparser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)


    cfg = get_config('config.ini')

    # model = models.googlenet(pretrained=True)
    # model.eval()
    # model.cuda()

    from model_wrapper import get_model_wrapper
    model = get_model_wrapper('GoogleNetWrapper')()
    model.model.cuda()

    mask = Mask(model, cfg.MASK)
    _, _, _, cam = mask(args.image_path)
    cv2.imwrite(os.path.join(args.save_path, 'cam.jpg'), cam)
