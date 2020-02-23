import os
import sys
import argparse
import cv2 as cv
import numpy as np
from glob import glob
from train.py import setup_gpus
import torch
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as psnr


def main():

    parser = argparse. ArgumentParser(description='Image Denoising')
    parser.add_argument('--img_dir', type=str, default='data/Set12', help='location of files to denoise')
    parser.add_argument('--img', type=str, help='location of a file to denoise')
    parser.add_argument('--noise_level', type=float, default=25.0, help='noise level for training')
    parser.add_argument('--model_name', type=str, default='logs/best_model.pt', help='location of model file')
    args = parser.parse_args()

    # normalize noise level
    noise_level = args.noise_level/255

    assert os.path.exists(args.img_dir), f'Image directory {args.img_dir} not found'
    assert os.path.exists(args.model_name), f'Model {args.model_name} not found'

    # load model 
    model = torch.load(args.model_name)
    model.eval() 

    if args.img_dir is not None:
        os.makedirs(os.path.join(args.img_dir, 'denoised'), exist_ok=True)

    test_psnr = 0
    num_test_files = 0
    for f in sorted(glob(os.path.join(args.img_dir, '*.png'))):
        print(f'Denoising {f}')

        # prepare clean image
        img = cv.imread(f).astype(np.float32)
        clean_img = img[:,:,0]/255
        clean_img = np.reshape(clean_img, (1,1,clean_img.shape[0], clean_img.shape[1]))
        clean_img = Variable(torch.Tensor(clean_img).cuda())

        # prepare noisy image
        noise = torch.FloatTensor(clean_img.size()).normal_(mean=0, std=noise_level)
        noisy_img = Variable((clean_img + noise).cuda())

        with torch.no_grad():
            denoised_img = torch.clap(noisy_img - model(noisy_img), 0.0, 1.0)
            
        # save images
        file_dir_name = os.path.dirname(f)
        file_name = os.path.basename(f)

        denoised_img = np.array(denoised_img.cpu().data.numpy.reshape((denoised_img.shape[2], denoised_img.shape[3], denoised_img.shape[1]))).astype(np.float32) 
        denoised_img *= 255     # undo normalization
        cv.imwrite(img=denoised_img.clip(0.0, 255.0).astype('uint8'), filename=os.path.join(file_dir_name, file_name.replace('.png', '-denoised.png')))

        noisy_img = np.array(noisy_img.cpu().data.numpy.reshape((denoised_img.shape[2], denoised_img.shape[1], denoised_img.shape[0])))
        noisy_img *= 255        # undo normalization
        cv.imwrite(img=noisy_img.clip(0.0, 255.0).astype('uint8'), filename=os.path.join(file_dir_name, file_name.replace('.png', '-noisy.png')))

        pnsr = psnr(img, denoised_img, data_range=255)
        print(f'PNSR of {f}: {pnsr}')

        test_pnsr += pnsr
        num_test_files += 1

    print(f'Average PNSR of testset is {test_pnsr/num_test_files}')

    return 0

if __name__ == '__main__':
    sys.exit(main())
