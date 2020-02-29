import os
import sys
import h5py
import numpy as np


def compare_datasets(dataset1, dataset2):
    with h5py.File(dataset1, 'r') as val:
        #print(val.keys())
        keys = list(val.keys())
        val_set = val[keys[0]]    
        print(val_set.shape)

def main():
    #dataset1 = sys.argv[1]
    #dataset2 = sys.argv[2]
    dataset1 = '/stash/tlab/mcarson/val.h5' 
    dataset2 = 'val-orig.h5' 
    
    print(f'Comparing dataset {dataset1} to {dataset2}')
    compare_datasets(dataset1, dataset2)


if __name__ == '__main__':
    
    sys.exit(main())