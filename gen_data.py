import os
import sys
import numpy as np
import argparse
import cv2 as cv
import h5py
from glob import glob



def generate_data(train_path, valid_path, patch_size, stride, scaling_factors):
    print(f'[Data Generation] Creating training data from {train_path}')
    num_train = 0
    h5f = h5py.File('train.h5', 'w')
    for f in sorted(glob(os.path.join(train_path, '*.png'))):
        print(f'Loading {f}')
        img = cv.imread(f)
        height, width, ch = img.shape

        for scale in scaling_factors:
            img = cv.resize(img, (int(height*scale), int(width*scale)), interpolation=cv.INTER_CUBIC)
            img = np.array(np.expand_dims(img[:,:,0]/255, axis=0), dtype=float)


    print(f'[Data Generation] Creating validation data from {valid_path}')
    num_valid = 0
    h5f = h5py.File('valid.h5', 'w')
    for f in sorted(glob(os.path.join(valid_path, '*.png'))):
        print(f'Loading {f}')
        img = cv.imread(f)
        img = np.array(np.expand_dims(img[:,:,0], axis=0), dtype=float)
        h5f.create_dataset(str(num_valid), data=img)
        num_valid += 1
    h5f.close()
        
    print(f'Number of training examples {num_train}')    
    print(f'Number of validation examples {num_valid}')    


    pass

def main():

    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    parser = argparse.ArgumentParser(description="DnCNN-data generation")
    parser.add_argument("--train_path", type=str, default='data/train', help='root directory for training data')
    parser.add_argument("--valid_path", type=str, default='data/Set12', help='root directory for validation data')
    parser.add_argument("--patch_size", type=int, default=40, help="image patch size to train on")
    parser.add_argument("--stride", type=int, default=10, help="image patch stride")
    parser.add_argument("--scaling_factors", type=str, default='1,.9,.8,.7', help="image scaling")
    args = parser.parse_args()

    train_path = os.path.join(script_dir, args.train_path)
    valid_path = os.path.join(script_dir, args.valid_path)
    patch_size = args.patch_size
    stride = args.stride
    scaling_factors = [float(scale) for scale in args.scaling_factors.split(',')] 

    print(f'[args] training data: {train_path}')
    print(f'[args] validation data: {valid_path}')
    print(f'[args] patch size: {patch_size}, stride: {stride}')
    print(f'[args] scaling factors: {scaling_factors}')

    generate_data(train_path, valid_path, patch_size, stride, scaling_factors)

if __name__ == '__main__':
    sys.exit(main())
