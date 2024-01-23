import hydra
from omegaconf import DictConfig
from dataset import get_ds
import os
import numpy as np

def tfds_to_np(tfds):
    X = []
    Y = []
    for x, y in tfds.as_numpy_iterator():
        X.append(x.reshape(1,8,8,2))
        Y.append(y)
    return np.concatenate(X), np.vstack(Y)

@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    train, test = get_ds(cfg, '../../data/st_handpose/ST_VL53L8CX_handposture_dataset', shuffle=True, split=0.8)

    x_train, y_train = tfds_to_np(train)
    x_test, y_test = tfds_to_np(test)
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # print (x_train.shape, y_train.shape)
    # print (x_test.shape, y_test.shape)

    dir = '../../data/st_handpose/generated_dataset'
    os.makedirs(dir, exist_ok=True)
    np.save(f'{dir}/x_train.npy', x_train)
    np.save(f'{dir}/y_train.npy', y_train)
    np.save(f'{dir}/x_test.npy', x_test)
    np.save(f'{dir}/y_test.npy', y_test)

if __name__ == '__main__':
    main()

