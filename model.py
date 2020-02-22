import torch
import torch.nn as nn

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
        # create sequential model
        #self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        #preds = self.model(x)
        for layer in self.model:
            x = layer(x)
        return x 




        