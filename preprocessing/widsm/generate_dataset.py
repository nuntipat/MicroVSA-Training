import hydra
from omegaconf import DictConfig
from dataset import WISDM
import os
import numpy as np

@hydra.main(version_base=None, config_path="", config_name="user_config")
def main(cfg: DictConfig) -> None:
    data_handler = WISDM(cfg)
    data_handler.prepare_data()

    # print(data_handler.train_x.shape)
    # print (data_handler.train_y.shape)
    # print(data_handler.validation_x.shape)
    # print(data_handler.validation_y.shape)
    # print(data_handler.test_x.shape)
    # print(data_handler.test_y.shape)

    dir = '../../data/wisdm/generated_dataset'
    os.makedirs(dir, exist_ok=True)
    np.save(f'{dir}/x_train.npy', data_handler.train_x)
    np.save(f'{dir}/y_train.npy', data_handler.train_y)
    np.save(f'{dir}/x_val.npy', data_handler.validation_x)
    np.save(f'{dir}/y_val.npy', data_handler.validation_y)
    np.save(f'{dir}/x_test.npy', data_handler.test_x)
    np.save(f'{dir}/y_test.npy', data_handler.test_y)

if __name__ == '__main__':
    main()

