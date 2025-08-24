import pandas as pd
import numpy as np

def get_isolet():
    test_data = pd.read_csv('../../data/isolet/isolet5.data', delim_whitespace=False, header=None)
    train_data = pd.read_csv('../../data/isolet/isolet1+2+3+4.data', delim_whitespace=False, header=None)

    x_test = test_data.values[:, :-1]
    y_test = test_data.values[:, -1]
    y_test -= 1
    x_train = train_data.values[:, :-1]
    y_train = train_data.values[:, -1]
    y_train -= 1

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    # shift the range from -1 to 1 to 0 - 1
    x_train = (x_train + 1) / 2
    x_test = (x_test + 1) / 2
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)
    y_train = np.array(y_train, dtype='uint8').flatten()
    y_test = np.array(y_test, dtype='uint8').flatten()

    np.save('../../data/isolet/x_train.npy', x_train)
    np.save('../../data/isolet/x_test.npy', x_test)
    np.save('../../data/isolet/y_train.npy', y_train)
    np.save('../../data/isolet/y_test.npy', y_test)

if __name__ == '__main__':
    get_isolet()