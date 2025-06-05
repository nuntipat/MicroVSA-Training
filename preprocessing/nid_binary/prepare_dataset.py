import os
from urllib.request import urlretrieve
import numpy as np

def get_nid_dataset():
    dataset_path = '../../data/nid/unsw_nb15_binarized.npz'
    if not os.path.exists(dataset_path):
        urlretrieve('https://zenodo.org/records/4519767/files/unsw_nb15_binarized.npz?download=1', dataset_path)

    unsw_nb15_data = np.load(dataset_path)

    x_train = unsw_nb15_data['train'][:, :-1].astype(np.float32)
    y_train = unsw_nb15_data['train'][:, -1]
    x_test = unsw_nb15_data['test'][:, :-1].astype(np.float32)
    y_test = unsw_nb15_data['test'][:, -1]

    np.save('../../data/nid/x_train.npy', x_train)
    np.save('../../data/nid/x_test.npy', x_test)
    np.save('../../data/nid/y_train.npy', y_train)
    np.save('../../data/nid/y_test.npy', y_test)

if __name__ == '__main__':
    get_nid_dataset()