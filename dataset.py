import numpy as np
import pandas as pd
import torchvision

rng = np.random.default_rng()


def print_stats(x_train, y_train, x_test, y_test):
    num_class = np.unique(y_train).shape[0]

    print(f'X Train shape : {x_train.shape}')
    print(f'X Range : {np.min(x_train)} {np.max(x_train)}')
    print(f'Y Train shape : {y_train.shape}')
    for i in range(num_class):
        print(f'    class {i} : {x_train[y_train == i].shape[0]}')

    print(f'X Test shape : {x_test.shape}')
    print(f'Y Test shape : {y_test.shape}')
    for i in range(num_class):
        print(f'    class {i} : {x_test[y_test == i].shape[0]}')

def get_mnist(show_stats=False):
    train_set = torchvision.datasets.MNIST(
        root='./data/mnist/',
        train=True,
        download=False
    )

    test_set = torchvision.datasets.MNIST(
        root='./data/mnist/',
        train=False,
        download=False
    )
    
    x_train = train_set.data.numpy().astype(np.float32)
    x_train = x_train.reshape((x_train.shape[0], -1))

    x_test = test_set.data.numpy().astype(np.float32)
    x_test = x_test.reshape((x_test.shape[0], -1))
    
    y_train = train_set.targets.numpy()
    y_test = test_set.targets.numpy()

    if show_stats:
        print_stats(x_train, y_train, x_test, y_test)
    return (x_train, y_train, x_test, y_test)

def get_ptb_ecg(show_stats=False):
    normal = np.loadtxt('data/ptb_ecg/ptbdb_normal.csv', delimiter=',')
    abnormal = np.loadtxt('data/ptb_ecg/ptbdb_abnormal.csv', delimiter=',')

    combine = np.concatenate([normal, abnormal])
    combine = combine[rng.permutation(combine.shape[0])]

    train_count = int(combine.shape[0] * 0.8)
    train = combine[:train_count]
    x_train = train[:, :-1].astype(np.float32)
    y_train = train[:, -1:].flatten().astype(int)
    test = combine[train_count:]
    x_test = test[:, :-1].astype(np.float32)
    y_test = test[:, -1:].flatten().astype(int)

    if show_stats:
        print_stats(x_train, y_train, x_test, y_test)
    return x_train, y_train, x_test, y_test


def get_ucihar(show_stats=False):
    test_data = pd.read_csv('data/ucihar/X_test.txt', sep='\s+', header=None)
    test_label = pd.read_csv('data/ucihar/y_test.txt', sep='\s+', header=None)
    train_data = pd.read_csv('data/ucihar/X_train.txt', sep='\s+', header=None)
    train_label = pd.read_csv('data/ucihar/y_train.txt', sep='\s+', header=None)
    
    x_test = test_data.values
    y_test = test_label.values
    y_test -= 1

    x_train = train_data.values
    y_train = train_label.values
    y_train -= 1

    # quantize the value to 256 levels
    x_train = np.round((x_train + 1) / 2 * 255)
    x_test = np.round((x_test + 1) / 2 * 255)

    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype='int').flatten()
    y_test = np.array(y_test, dtype='int').flatten()

    if show_stats:
        print_stats(x_train, y_train, x_test, y_test)
    return x_train, y_train, x_test, y_test


def get_qksd(selected_word='hey_snapdragon', show_stats=False): # selected_word can be hey_android, hey_snapdragon, hi_galaxy, hi_lumina
    x_train = np.load(f'data/qksd/{selected_word}/x_train.npy').astype(np.float32)
    y_train = np.load(f'data/qksd/{selected_word}/y_train.npy').flatten().astype(int)
    x_test = np.load(f'data/qksd/{selected_word}/x_test.npy').astype(np.float32)
    y_test = np.load(f'data/qksd/{selected_word}/y_test.npy').flatten().astype(int)

    permute_index = rng.permutation(x_train.shape[0])
    x_train = x_train[permute_index]
    y_train = y_train[permute_index]

    if show_stats:
        print_stats(x_train, y_train, x_test, y_test)
    return x_train, y_train, x_test, y_test


def get_fsdd(show_stats=False):
    x_train = np.load(f'data/fsdd/x_train.npy').astype(np.float32)
    y_train = np.load(f'data/fsdd/y_train.npy').flatten().astype(int)
    x_test = np.load(f'data/fsdd/x_test.npy').astype(np.float32)
    y_test = np.load(f'data/fsdd/y_test.npy').flatten().astype(int)

    if show_stats:
        print_stats(x_train, y_train, x_test, y_test)
    return x_train, y_train, x_test, y_test


def get_wisdm(show_stats=False):
    x_train = np.load('data/wisdm/generated_dataset/x_train.npy').astype(np.float32)
    y_train = np.load('data/wisdm/generated_dataset/y_train.npy').astype(int)
    x_val = np.load('data/wisdm/generated_dataset/x_val.npy').astype(np.float32)
    y_val = np.load('data/wisdm/generated_dataset/y_val.npy').astype(int)
    x_test = np.load('data/wisdm/generated_dataset/x_test.npy').astype(np.float32)
    y_test = np.load('data/wisdm/generated_dataset/y_test.npy').astype(int)

    if show_stats:
        print_stats(x_train, y_train, x_test, y_test)
    return x_train, y_train, x_val, y_val, x_test, y_test


def get_st_handpose(show_stats=False):
    x_train = np.load('data/st_handpose/generated_dataset/x_train.npy').astype(np.float32)
    y_train = np.load('data/st_handpose/generated_dataset/y_train.npy').astype(int)
    x_test = np.load('data/st_handpose/generated_dataset/x_test.npy').astype(np.float32)
    y_test = np.load('data/st_handpose/generated_dataset/y_test.npy').astype(int)

    x_train = np.hstack([((x_train[:,:,:,0] + 1) / 20.0).reshape(x_train.shape[0], -1), ((x_train[:,:,:,1] + 1) / 16.0).reshape(x_train.shape[0], -1)])
    x_test = np.hstack([((x_test[:,:,:,0] + 1) / 20.0).reshape(x_test.shape[0], -1), ((x_test[:,:,:,1] + 1) / 16.0).reshape(x_test.shape[0], -1)])

    if show_stats:
        print_stats(x_train, y_train, x_test, y_test)
    return x_train, y_train, x_test, y_test

def get_jsc(show_stats=False):
    x_train = np.load(f'data/jsc/x_train.npy').astype(np.float32)
    y_train = np.load(f'data/jsc/y_train.npy').flatten().astype(int)
    x_test = np.load(f'data/jsc/x_test.npy').astype(np.float32)
    y_test = np.load(f'data/jsc/y_test.npy').flatten().astype(int)

    permute_index = np.random.permutation(x_train.shape[0])
    x_train = x_train[permute_index]
    y_train = y_train[permute_index]

    if show_stats:
        print_stats(x_train, y_train, x_test, y_test)
    return x_train, y_train, x_test, y_test

def get_nid(show_stats=False):
    x_train = np.load(f'data/nid/x_train.npy').astype(np.float32)
    y_train = np.load(f'data/nid/y_train.npy').flatten().astype(int)
    x_test = np.load(f'data/nid/x_test.npy').astype(np.float32)
    y_test = np.load(f'data/nid/y_test.npy').flatten().astype(int)

    permute_index = np.random.permutation(x_train.shape[0])
    x_train = x_train[permute_index]
    y_train = y_train[permute_index]

    if show_stats:
        print_stats(x_train, y_train, x_test, y_test)
    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    get_jsc(True)