import numpy as np
import os
import sys
import cv2 as cv
import torch
from glob import glob
from gen_data import image_augment
from gen_data import augment_img

from train import gen_noise, downsample
from skimage.metrics import peak_signal_noise_ratio as psnr


def main():
    noise_types = np.array(['normal', 'uniform', 's&p'])

    img_dir = 'data/train_color/test'
    imgs =  sorted(glob(os.path.join(img_dir, '*.png')))
    assert len(imgs) > 0, f'No images found in {img_dir}'
    img = cv.imread(imgs[0])
    images = np.empty((len(imgs), img.shape[2], img.shape[0], img.shape[1]))
    #images = np.empty((len(imgs), img.shape[0], img.shape[1], img.shape[2]))
    
    aug_types = np.array([1.0, .9, .8, .7])
    for i, imgF in enumerate(imgs):
        img = cv.imread(imgF)/255
        cv.imshow('original', (img*255).clip(0,255).astype('uint8'))
        for aug in aug_types:
            img_aug = cv.resize(img, (int(img.shape[1]*aug), int(img.shape[0]*aug)), interpolation=cv.INTER_CUBIC)
            cv.imshow(f'scale factor {aug}', (img_aug*255).clip(0,255).astype('uint8'))
            out_path = os.path.join('augs', os.path.basename(imgF).replace('.png', f'-{aug}.png')) 
            cv.imwrite(out_path, (img_aug*255).clip(0,255).astype('uint8'))
        cv.waitKey(0)

    aug_types = np.array(['mirror', 'flip', 'rotL', 'rotR'])
    for i, imgF in enumerate(imgs):
        img = cv.imread(imgF)/255
        cv.imshow('original', (img*255).clip(0,255).astype('uint8'))
        for aug in aug_types:
            img_aug = augment_img(img, aug)
            cv.imshow(aug, (img_aug*255).clip(0,255).astype('uint8'))
            #out_path = os.path.join('augs', os.path.basename(imgF).replace('.png', f'-{aug}.png')) 
            #cv.imwrite(out_path, (img_aug*255).clip(0,255).astype('uint8'))
        cv.waitKey(0)

    for i,img in enumerate(imgs):
        img = cv.imread(img)/255


    for i,img in enumerate(imgs):
        img = cv.imread(img)/255
        img = np.einsum('ijk->kij', img.astype(np.float32)) 
        images[i,:,:,:] = img 

    scales = [2,3,4]
    for scale in scales:
        downsample_imgs, diff_imgs = downsample(images, scales)

        for i in range(downsample_imgs.shape[0]):
            clean_img = np.einsum('ijk->jki', images[i].astype(np.float32)) 
            cv.imshow('orig', (clean_img*255).clip(0,255).astype('uint8'))
            down_img = np.einsum('ijk->jki', downsample_imgs[i].astype(np.float32)) 
            cv.imshow('downsampled', (down_img*255).clip(0,255).astype('uint8'))
            diff = np.einsum('ijk->jki', diff_imgs[i].astype(np.float32)) 
            cv.imshow('diff', (diff*255).clip(0,255).astype('uint8'))
            recon = down_img - diff 
            cv.imshow('recons', (recon*255).clip(0,255).astype('uint8'))
            cv.waitKey(0)
        

    images = torch.FloatTensor(images)
    for nt in noise_types:
        noise = gen_noise(images.size(), nt)
        noisy_images = images + noise.data.numpy()

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
