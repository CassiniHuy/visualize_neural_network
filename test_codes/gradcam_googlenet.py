import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model._modules.items():
            if 'aux' in module_pos:
                continue
            x = module(x)  # Forward
            if module_pos == 'avgpool':
                x = torch.flatten(x, 1)
            if module_pos == 'inception5b':
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        return conv_output, x


class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.cpu().detach().numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).cuda().zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.cpu().detach().numpy()[0]
        # Get convolution outputs
        target = conv_output.cpu().detach().numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        # cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = cam - np.min(cam)  # Normalize between 0-1
        cam = cam / np.max(cam)  # Normalize between 0-1
        cam = cv2.resize(cam, (224, 224))
        return np.uint8(cam * 255)

totensor_and_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

def load_input(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img_tensor = totensor_and_normalize(img)[None, :,:,:]
    return img_tensor

if __name__ == '__main__':
    import os
    from argparse import ArgumentParser
    argparser = ArgumentParser()
    argparser.add_argument('--image_path', type=str)
    argparser.add_argument('--mask_path', type=str)
    args = argparser.parse_args()
    imgs = os.listdir(args.image_path)
    print('images:', len(imgs))
    pretrained_model = torchvision.models.googlenet(pretrained=True, transform_input=True)
    pretrained_model.eval()
    pretrained_model.cuda()
    # Grad cam
    grad_cam = GradCam(pretrained_model, target_layer=11)

    for img in imgs:
        prep_img = load_input(os.path.join(args.image_path, img)).cuda()
        cam = grad_cam.generate_cam(prep_img)
        # Save mask
        cv2.imwrite(os.path.join(args.mask_path, img), cam)
        print(img, 'done')
