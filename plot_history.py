import os
import sys
import pickle
import matplotlib.pyplot as plt


def plt_losses(train_loss, val_loss, sav_dir):

    plt.figure()
    plt.plot(train_loss, label='training loss', color='blue')
    plt.plot(val_loss, label='validation loss', color='red')

    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(sav_dir, 'losses.png'))
    plt.show()

def plt_psnrs(psnrs, sav_dir):

    plt.figure()
    for key in psnrs.keys():
        plt.plot(psnrs[key], label=f'PSNR {key} noise')

    plt.ylabel('PSNR')
    plt.xlabel('epoch')

    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(sav_dir, 'psnrs.png'))
    plt.show()

def main():
    history_file = sys.argv[1]
    print(f'Creating plots with history file {history_file}')

    data = pickle.load(open(history_file, 'rb'))

    plt_losses(data['train'], data['val'], os.path.dirname(history_file))
    plt_psnrs(data['psnr'], os.path.dirname(history_file))

    return 0

if __name__ == '__main__':
    
    sys.exit(main())
