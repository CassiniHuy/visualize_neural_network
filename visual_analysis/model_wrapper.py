import os
import torch
import torchvision
import torch.nn.functional as F
from abc import ABCMeta
from abc import abstractmethod
import numpy as np
import gc
import os
import sys

class ModelWrapper(object):
    ''' Interface for Adversarial Defense '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.model = None
        self.model_name = None
        self.constraint = {}

    @abstractmethod
    def if_apply_constraint(self, layer_name):
        pass

    @abstractmethod
    def set_constraint(self, x):
        pass

    def get_save_value_hook(self, layer_name):
        def save_value_hook(mod, ipt, opt):
            out = opt.detach()
            self.constraint[layer_name + '_bu'] = torch.max(out, torch.zeros_like(out))
            self.constraint[layer_name + '_bl'] = torch.min(out, torch.zeros_like(out))
        return save_value_hook

    def get_constraint_hook(self, layer_name):
        def constraint_hook(mod, ipt, opt):
            output = torch.min(
                self.constraint[layer_name + '_bu'], 
                torch.max(self.constraint[layer_name + '_bl'], opt))
            return output
        return constraint_hook

class GoogleNetWrapper(ModelWrapper, torch.nn.Module):
    def __init__(self):
        ModelWrapper.__init__(self)
        torch.nn.Module.__init__(self)
        self.model = torchvision.models.googlenet(pretrained=True, transform_input=True)
        self.model.eval()
        self.model_name = 'googlenet_public'
    
    def if_apply_constraint(self, layer_name):
        if 'conv' in layer_name or 'inception' in layer_name:
            return True
        else:
            return False
    
    def set_constraint(self, x):
        # Get original input activations
        handles = []
        for name, layer in self.model._modules.items():
            if self.if_apply_constraint(name):
                handle = layer.register_forward_hook(self.get_save_value_hook(name))
                handles.append(handle)
        self.model(x)
        for handle in handles:
            handle.remove()
        # Set hook after every nonlinear layer
        for name, layer in self.model._modules.items():
            if self.if_apply_constraint(name):
                layer.register_forward_hook(self.get_constraint_hook(name))

    def forward(self, x):
        return self.model(x)

class AlexNetWrapper(ModelWrapper, torch.nn.Module):
    def __init__(self):
        ModelWrapper.__init__(self)
        torch.nn.Module.__init__(self)
        self.model = torchvision.models.alexnet(pretrained=True)
        self.model.eval()
        self.model_name = 'alexnet_public'
    
    def if_apply_constraint(self, layer_name):
        if 'relu' in layer_name.lower():
            return True
        else:
            return False
    
    def set_constraint(self, x):
        # Get original input activations
        handles = []
        for name, layer in self.model.features._modules.items():
            if self.if_apply_constraint(str(layer)):
                handle = layer.register_forward_hook(self.get_save_value_hook('features_' + name))
                handles.append(handle)
        for name, layer in self.model.classifier._modules.items():
            if self.if_apply_constraint(str(layer)):
                handle = layer.register_forward_hook(self.get_save_value_hook('classifier_' + name))
                handles.append(handle)
        self.model(x)
        for handle in handles:
            handle.remove()
        # Set hook after every ReLU layer
        for name, layer in self.model.features._modules.items():
            if self.if_apply_constraint(str(layer)):
                handle = layer.register_forward_hook(self.get_constraint_hook('features_' + name))
                handles.append(handle)
        for name, layer in self.model.classifier._modules.items():
            if self.if_apply_constraint(str(layer)):
                handle = layer.register_forward_hook(self.get_constraint_hook('classifier_' + name))
                handles.append(handle)

    def forward(self, x):
        return self.model(x)

def get_model_wrapper(name):
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name)
