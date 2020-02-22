import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import h5py
from model import DnCNN
from utils import weights_init_kaiming
from tqdm import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_name):
        super(Dataset, self).__init__()
        self.file_name = file_name
        with h5py.File(file_name, 'r') as data:
            self.keys = list(data.keys())

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        with h5py.File(self.file_name, 'r') as data:
            example = np.array(data[self.keys[index]])
        return torch.Tensor(example)

    def shape(self):
        with h5py.File(self.file_name, 'r') as data:
            return np.array(data[self.keys[0]]).shape

        
def setup_gpus():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device_ids = [i for i in range(torch.cuda.device_count())]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))
    return device_ids

def main():

    parser = argparse. ArgumentParser(description='Image Denoising Trainer')
    parser.add_argument('--train_set', type=str, default='train.h5', help='h5 file with training vectors')
    parser.add_argument('--val_set', type=str, default='val.h5', help='h5 file with validation vectors')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--num_layers', type=int, default=17, help='number of CNN layers in network')
    parser.add_argument('--num_filters', type=int, default=64, help='number of filters per CNN layer')
    parser.add_argument('--filter_size', type=int, default=3, help='size of filter for CNN layers')
    parser.add_argument('--stride', type=int, default=1, help='filter stride for CNN layers')
    parser.add_argument('--noise_level', type=float, default=25.0, help='noise level for training')
    args = parser.parse_args()

    # noise level for training, must be normalized like the clean image
    noise_level = args.noise_level/255

    assert os.path.exists(args.train_set), f'Cannot find training vectors file {args.train_set}'
    assert os.path.exists(args.val_set), f'Cannot find validation vectors file {args.train_set}'


    print('Loading datasets')

    train_data = Dataset(args.train_set)
    val_data = Dataset(args.val_set)

    print(f'Number of training examples: {len(train_data)}')
    print(f'Number of validation examples: {len(val_data)}')

    train_loader = DataLoader(dataset=train_data, num_workers=os.cpu_count(), batch_size=args.batch_size, shuffle=True)

    # input shape for each example to network, NOTE: channels first
    num_channels, patch_height, patch_width = train_data.shape()

    print(f'Input shape to model forward will be: ({args.batch_size}, {num_channels}, {patch_height}, {patch_width})')

    # detect gpus and setup environment variables
    device_ids = setup_gpus()
    print(f'Cuda devices found: {[torch.cuda.get_device_name(i) for i in device_ids]}')

    # create model
    model = DnCNN(num_channels=num_channels, patch_size=patch_height,  num_layers=args.num_layers, \
                  kernel_size=args.filter_size, stride=args.stride, num_filters=args.num_filters) 
    model.apply(weights_init_kaiming)

    # move model to available gpus
    model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()

    # setup loss and optimizer
    criterion = torch.nn.MSELoss(reduction='sum').cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # data struct to track training and validation losses per epoch
    history = {'train':[], 'val':[]}

    # Main training loop
    for epoch in range(args.epochs):
        print(f'Staring epoch {epoch+1}')

        # iterate through batches of training examples
        epoch_train_loss = 0
        epoch_val_loss = 0
        steps = 0
        for data in tqdm(train_loader):
            # generate additive white noise from gaussian distribution
            noise = torch.FloatTensor(data.size()).normal_(mean=0, std=noise_level)

            # pack input and target it into torch variable
            examples = Variable((data + noise).cuda()) 
            noise = Variable(noise.cuda())

            # make predictions
            preds = model(examples)

            # calculate loss
            loss = criterion(preds, noise)/examples.size()[0]
            epoch_train_loss += loss

            # backprop
            loss.backward()
            optimizer.step()

            steps += 1
        
        history['train'].append(epoch_train_loss/steps)
        print(f'Training loss: {epoch_train_loss/steps}')

    return 0

if __name__ == '__main__':
    sys.exit(main())
