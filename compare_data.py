import os
import sys
import h5py
import numpy as np


def compare_datasets(dataset1, dataset2):
    with h5py.File(dataset1, 'r') as val:
        #val_set = val[0]    
        #print(f'num keys {len(val.keys())}, keys: {val.keys()}')
        data = np.empty((len(list(val.keys())), 1, 40, 40))
        for i, key in enumerate(list(val.keys())):
            a = val[key]
            #b = np.expand_dims(np.array(a), axis=0)
            data[i] = np.array(a)

def main():
    #dataset1 = sys.argv[1]
    #dataset2 = sys.argv[2]
    dataset1 = 'train.h5' 
    dataset2 = 'val-orig.h5' 
    
    print(f'Comparing dataset {dataset1} to {dataset2}')
    compare_datasets(dataset1, dataset2)


if __name__ == '__main__':
    
    sys.exit(main())