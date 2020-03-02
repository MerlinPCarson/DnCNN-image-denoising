import os
import sys
import numpy as np
import argparse
import cv2 as cv
import h5py
from glob import glob
from tqdm import tqdm

#import matplotlib.pyplot as plt

def augment_img(img, aug_type):
    if aug_type == 'mirror':
        return np.fliplr(img)
    if aug_type == 'flip':
        return np.flipud(img)
    if aug_type == 'rotL':
        return np.rot90(img, k=1)
    if aug_type == 'rotR':
        return np.rot90(img, k=-1)


def image_augment(img, num_augs):
    # save as channel first
    data_aug = np.empty((num_augs+1,img.shape[2], img.shape[0], img.shape[1]))
    aug_types = np.array(['mirror', 'flip', 'rotL', 'rotR'])
    np.random.shuffle(aug_types)
    data_aug[0] = np.einsum('ijk->kij', img.astype(np.float32)) 
    for i in range(num_augs):
        data_aug[i+1] = np.einsum('ijk->kij', augment_img(img, aug_types[i]).astype(np.float32)) 

    return data_aug

def downsample(clean_imgs, scale):
    downsample_imgs = np.empty(clean_imgs.shape)
    height, width, _ = clean_imgs[0].shape
    down_h = int(height/scale)
    down_w = int(width/scale)
    for i in range(clean_imgs.shape[0]):
        resized_img = cv.resize(clean_imgs[i], (down_w,down_h), cv.INTER_AREA)
        downsample_imgs[i] = cv.resize(resized_img, (width,height), cv.INTER_AREA)
    return downsample_imgs

def generate_data(train_path, val_path, test_path, patch_size, stride, scaling_factors, num_augments, num_channels):
    #num_channels = 3
    scales = [2,3,4]
    print(f'[Data Generation] Creating training data from {train_path} with {num_channels} channels')
    num_train = 0
    h5f = h5py.File('/stash/tlab/mcarson/train.h5', 'w')
    num_train = 0
    for f in tqdm(sorted(glob(os.path.join(train_path, '*.png')))):
        #print(f'{num_train+1}: Preprocessing {f}')
        img = cv.imread(f)
        height, width, ch = img.shape

        for scale in scaling_factors:
            img_scaled = cv.resize(img, (int(height*scale), int(width*scale)), interpolation=cv.INTER_CUBIC)
            img_scaled = np.array(img_scaled[:,:,:num_channels].reshape((img_scaled.shape[0],img_scaled.shape[1],num_channels))/255)
            patches = get_image_patches(img_scaled, patch_size, stride)
            #print(f'  scaling: {scale}, num patches: {patches.shape[0]}')
            for patch_num in range(patches.shape[0]):
                data_aug = image_augment(patches[patch_num], num_augments)
                for aug in range(data_aug.shape[0]):
                    h5f.create_dataset(str(num_train), data=data_aug[aug])
                    num_train += 1
                # downsampling
                for scale in scales:
                    downsampled_imgs = downsample(data_aug)
                    for i in range(downsampled_imgs.shape[0]):
                        h5f.create_dataset(str(num_train), data=downsampled_imgs[i])
                        num_train += 1

    h5f.close()

    print(f'[Data Generation] Creating validation data from {val_path}')
    num_val = 0
    h5f = h5py.File('/stash/tlab/mcarsonval.h5', 'w')
    for f in tqdm(sorted(glob(os.path.join(val_path, '*.png')))):
        #print(f'Preprocessing {f}')
        img = cv.imread(f)
        img = np.array(img[:,:,:num_channels].reshape((img.shape[0],img.shape[1],num_channels))/255)
        patches = get_image_patches(img, patch_size, stride)
        for patch_num in range(patches.shape[0]):
            # channels first
            patch = np.einsum('ijk->kij', patches[patch_num].astype(np.float32)) 
            h5f.create_dataset(str(num_val), data=patch)
            num_val += 1

    h5f.close()

    num_sisr_val = 0
    h5f = h5py.File('/stash/tlab/mcarson/val_sisr.h5', 'w')
    # downsampling
    for scale in scales:
        downsampled_imgs = downsample(patches, scale)
        for i in range(downsampled_imgs.shape[0]):
            d_img = np.einsum('ijk->kij', downsampled_imgs[i].astype(np.float32)) 
            h5f.create_dataset(str(num_sisr_val), data=d_img)
            num_sisr_val += 1
        
    h5f.close()

#    print(f'[Data Generation] Creating test data from {test_path}')
#    num_test = 0
#    h5f = h5py.File('test.h5', 'w')
#    for f in sorted(glob(os.path.join(test_path, '*.png'))):
#        print(f'Preprocessing {f}')
#        img = cv.imread(f)
#        # channels first
#        img = np.array(img[:,:,0].reshape((1,img.shape[0],img.shape[1]))/255, dtype=np.float32)
#        h5f.create_dataset(str(num_test), data=img)
#        num_test += 1
#    h5f.close()
        
    print(f'Number of training examples {num_train}')    
    print(f'Number of validation examples {num_val}')    
    print(f'Number of validation examples SISR {num_sisr_val}')    

#    print(f'Number of test examples {num_test}')    


    pass

def get_image_patches(img, patch_size, stride):
    win_row_end = img.shape[0] - patch_size
    win_col_end = img.shape[1] - patch_size
    num_patches_rows = int((img.shape[0]-patch_size)/stride + 1)
    num_patches_cols = int((img.shape[1]-patch_size)/stride + 1)
    num_chs = int(img.shape[2])
    total_patches = int(num_patches_rows * num_patches_cols)

    patches = np.zeros((total_patches, patch_size, patch_size, num_chs), dtype=float)

    rows = np.arange(0,win_row_end+1, stride)
    cols = np.arange(0,win_col_end+1, stride)
    patch_num = 0
    for row in rows:
        for col in cols: 
            patch = img[row:row+patch_size, col:col+patch_size,:]
            patches[patch_num,:,:,:] = patch
            patch_num += 1

    return patches

def main():

    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    parser = argparse.ArgumentParser(description="DnCNN-data generation")
    parser.add_argument("--train_path", type=str, default='data/train_color/train', help='root directory for training data')
    parser.add_argument("--val_path", type=str, default='data/train_color/val', help='root directory for validation data')
    parser.add_argument("--test_path", type=str, default='data/train_color/test', help='root directory for test data')
    parser.add_argument("--patch_size", type=int, default=50, help="image patch size to train on")
    parser.add_argument("--stride", type=int, default=10, help="image patch stride")
    parser.add_argument("--scaling_factors", type=str, default='1,.6,.4,.2', help="image scaling")
    parser.add_argument("--num_augments", type=int, default=0, help="number of data augmentations per patch")
    parser.add_argument("--num_channels", type=int, default=3, help="number of channels (bw=1, color=3)")
    args = parser.parse_args()

    train_path = os.path.join(script_dir, args.train_path)
    val_path = os.path.join(script_dir, args.val_path)
    test_path = os.path.join(script_dir, args.test_path)
    patch_size = args.patch_size
    stride = args.stride
    scaling_factors = [float(scale) for scale in args.scaling_factors.split(',')] 
    num_augments = args.num_augments

    print(f'[args] training data: {train_path}')
    print(f'[args] validation data: {val_path}')
    print(f'[args] patch size: {patch_size}, stride: {stride}')
    print(f'[args] scaling factors: {scaling_factors}')
    print(f'[args] number of augmentations: {num_augments}')

    generate_data(train_path, val_path, test_path, patch_size, stride, scaling_factors, num_augments, args.num_channels)

    return 0

if __name__ == '__main__':
    sys.exit(main())
