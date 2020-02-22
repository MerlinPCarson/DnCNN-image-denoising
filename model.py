import torch
import torch.nn as nn
import numpy as np
import math

class DnCNN(nn.Module):
    def __init__(self, num_channels, patch_size, num_layers, kernel_size, stride, num_filters):
        super(DnCNN, self).__init__()

        padding = int((stride * (patch_size - 1) - patch_size + kernel_size)/2)
        print(f'Calculated padding needed to maintain image size: {padding}')

        # create module list
        self.layers = []
        self.layers.append(nn.Conv2d(in_channels=num_channels, out_channels=num_filters, kernel_size=kernel_size, padding=padding, bias=False))
        self.layers.append(nn.ReLU(inplace=True))
        for i in range(num_layers-2):
            self.layers.append(nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding=padding, bias=False))
            self.layers.append(nn.BatchNorm2d(num_filters))
            self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Conv2d(in_channels=num_filters, out_channels=num_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.model = nn.ModuleList(self.layers)
        self.init_weights()
        # create sequential model
        #self.model = nn.Sequential(*self.layers)

    def init_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight.data, a=0, mode='fan_in')
            if isinstance(layer, nn.BatchNorm2d):
                layer.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
                nn.init.constant_(layer.bias.data, 0.0)
                
    def forward(self, x):
        #preds = self.model(x)
        for layer in self.model:
            x = layer(x)
        return x 




        