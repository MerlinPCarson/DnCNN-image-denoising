import os
import sys
import h5py
import numpy as np


def compare_datasets(dataset1, dataset2):
    
    with h5py.File(dataset1, 'r') as ds1:
        print(ds1.keys())
        keys = ds1.keys()
        data1 = np.array(ds1['0'])
    with h5py.File(dataset2, 'r') as ds2:
        print(ds2.keys())
        keys = ds2.keys()
        data2 = np.array(ds2['0'])

        diff = data1-data2

        print(f'diff {np.sum(diff)}')

def main():
    #dataset1 = sys.argv[1]
    #dataset2 = sys.argv[2]
    dataset1 = 'val.h5' 
    dataset2 = 'val-orig.h5' 
    
    print(f'Comparing dataset {dataset1} to {dataset2}')
    compare_datasets(dataset1, dataset2)

    return 0

if __name__ == '__main__':
    
    sys.exit(main())