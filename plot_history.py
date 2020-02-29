import os
import sys
import pickle
import matplotlib.pyplot as plt


def plt_losses(train_loss, val_loss):

    plt.plot(train_loss, label='training loss', color='blue')
    plt.plot(val_loss, label='validation loss', color='red')

    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.legend()
    plt.tight_layout()

    plt.show()

def plt_psnrs(psnrs):

    for key in psnrs.keys():
        plt.plot(psnrs[key], label=f'PSNR {key} noise')

    plt.ylabel('PSNR')
    plt.xlabel('epoch')

    plt.legend()
    plt.tight_layout()

    plt.show()

def main():
    #history_file = sys.argv[1]
    history_file = 'logs/best_model.npy'
    print(history_file)

    data = pickle.load(open(history_file, 'rb'))

    plt_losses(data['train'], data['val'])
    plt_psnrs(data['psnr'])

    return 0

if __name__ == '__main__':
    
    sys.exit(main())