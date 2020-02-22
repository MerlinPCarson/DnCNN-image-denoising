import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from models import DnCNN
from dataset import prepare_data, Dataset
from utils import *
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device_ids = []
for i in range(torch.cuda.device_count()):
    device_ids.append(i)
device_str = ','.join(map(str, device_ids))
os.environ["CUDA_VISIBLE_DEVICES"] = device_str 

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=256, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
opt = parser.parse_args()


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    print("# of validation samples: %d\n" % int(len(dataset_val)))
    # Build model
 #   net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
 #   net.apply(weights_init_kaiming)
 #   criterion = nn.MSELoss(reduction='sum')
    # Move to GPU
    #device_ids = [0]
 #   model = nn.DataParallel(net, device_ids=device_ids).cuda()
    #print(device_ids)
    #model = nn.parallel.DistributedDataParallel(net, device_ids=device_ids).cuda()
    #model = nn.parallel.DistributedDataParallel(net).cuda()
 #   criterion.cuda()
    # Optimizer
 #   optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    #scheduler = ReduceLROnPlateau(optimizer, 'min')
    # training
 #   writer = SummaryWriter(opt.outf)
    step = 0
 #   noiseL_B=[0,55] # ingnored when opt.mode=='S'
    
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            lr = opt.lr
        else:
            lr = opt.lr / 10.
        # set learning rate
#        for param_group in optimizer.param_groups:
#            param_group["lr"] = lr
        # train
        print(f'Starting epoch {epoch+1} with lr={lr}')
        #for i, data in enumerate(loader_train, 0):
        for data in tqdm(loader_train):
            # training step
 #           model.train()
 #           model.zero_grad()
 #           optimizer.zero_grad()
            img_train = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
 #           noise = Variable(noise.cuda())
 #           out_train = model(imgn_train)
 #           loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
 #           loss.backward()
 #           optimizer.step()
            # results
 #           model.eval()
            #out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
            #psnr_train = batch_PSNR(out_train, img_train, 1.)
            #print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
            #    (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            #if step % 10 == 0:
            #    # Log the scalar values
            #    writer.add_scalar('loss', loss.item(), step)
            #    writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## the end of each epoch
 #       model.eval()
        # validate
        psnr_val = 0
        #1val_loss = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
            imgn_val = img_val + noise
 #           with torch.no_grad():
 #               img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
 #               out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
        #        val_loss += criterion(out_train, noise) / (imgn_val.size()[0]*2)
 #               psnr_val += batch_PSNR(out_val, img_val, 1.)
 #       psnr_val /= len(dataset_val)
        #val_loss /= len(dataset_val)
 #       print(f"\nPSNR_val: {psnr_val:.4f}")
        #print(f"\nval_loss: {val_loss:.4f}")
 #       writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        #writer.add_scalar('val_loss', val_loss.item(), epoch)
        # log the images
 #       out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
 #       Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
 #       Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
 #       Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
 #       writer.add_image('clean image', Img, epoch)
 #       writer.add_image('noisy image', Imgn, epoch)
 #       writer.add_image('reconstructed image', Irecon, epoch)
        # save model
 #       torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
