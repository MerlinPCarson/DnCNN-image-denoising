import numpy as np
import os
import sys
import cv2 as cv
import torch
from glob import glob

from train import gen_noise
from skimage.metrics import peak_signal_noise_ratio as psnr


def main():
    noise_types = np.array(['normal', 'uniform', 'pepper'])

    img_dir = 'data/train_color/test'
    imgs =  sorted(glob(os.path.join(img_dir, '*.png')))
    assert len(imgs) > 0, f'No images found in {img_dir}'
    img = cv.imread(imgs[0])
    images = np.empty((len(imgs), img.shape[2], img.shape[0], img.shape[1]))
    
    for i,img in enumerate(imgs):
        img = cv.imread(img)/255
        img = np.einsum('ijk->kij', img.astype(np.float32)) 
        images[i,:,:,:] = img 


    images = torch.FloatTensor(images)
    for nt in noise_types:
        noise = gen_noise(images.size(), nt)
        noisy_images = images + noise

        for i in range(images.size()[0]):
            psnr_val = psnr(images[i].data.numpy().astype(np.float32), noisy_images[i].data.numpy().astype(np.float32), data_range=1)
            print(f'PSNR:{psnr_val}')
            clean_img = images[i].data.numpy()*255
            clean_img = np.einsum('ijk->jki', clean_img.astype(np.float32)) 
            noisy_img = noisy_images[i].data.numpy()*255
            noisy_img = np.einsum('ijk->jki', noisy_img.astype(np.float32)) 
            noisy = noise.data.numpy()[i]*255
            noisy = np.einsum('ijk->jki', noisy.astype(np.float32)) 
            cv.imshow('clean image', clean_img.clip(0,255).astype('uint8'))
            cv.imshow('noisy image', noisy_img.clip(0,255).astype('uint8'))
            cv.imshow('noise image', noisy.clip(0,255).astype('uint8'))
            cv.waitKey(0)


    return 0

if __name__ == '__main__':
    sys.exit(main())
