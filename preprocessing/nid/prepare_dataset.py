import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, OrdinalEncoder

def get_nid_dataset_raw(only_categorical = True):
    train_path = '../../data/nid/UNSW_NB15_training-set.csv'
    test_path = '../../data/nid/UNSW_NB15_testing-set.csv'
    if (not os.path.exists(train_path)) or (not os.path.exists(test_path)):
        print ('Error: please download the dataset files (UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv) from https://research.unsw.edu.au/projects/unsw-nb15-dataset')
        sys.exit(1)

    train = pd.read_csv(train_path)
    train_feature = train.iloc[:, 1:-2]
    train_label = train['label']

    test = pd.read_csv(test_path)
    test_feature = test.iloc[:, 1:-2]
    test_label = test['label']

    categorical_col = ['proto', 'service', 'state']
    for col in categorical_col:
        e = OrdinalEncoder()
        e.fit(pd.concat([train_feature.loc[:, col:col], test_feature.loc[:, col:col]], axis=0))
        if e.categories_[0].shape[0] > 256:
            print ('Error: found more than 256 category')
            sys.exit(1)
        train_feature[col] = e.transform(train_feature.loc[:, col:col]) #/ e.categories_[0].shape[0]
        test_feature[col] = e.transform(test_feature.loc[:, col:col]) #/ e.categories_[0].shape[0]

    if not only_categorical:
        numeric_col = [col for col in train_feature.columns if col not in categorical_col]
        for col in numeric_col:
            e = QuantileTransformer(subsample=None)
            e.fit(train_feature.loc[:, col:col])
            # train_feature[col] = e.transform(train_feature.loc[:, col:col])
            # test_feature[col] = e.transform(test_feature.loc[:, col:col])
            train_feature[col] = e.transform(train_feature.loc[:, col:col]) * 255
            test_feature[col] = e.transform(test_feature.loc[:, col:col]) * 255

    if only_categorical:
        np.save('../../data/nid/x_train.npy', train_feature[categorical_col].to_numpy())
        np.save('../../data/nid/x_test.npy', test_feature[categorical_col].to_numpy())
    else:
        np.save('../../data/nid/x_train.npy', train_feature.to_numpy())
        np.save('../../data/nid/x_test.npy', test_feature.to_numpy())
    np.save('../../data/nid/y_train.npy', train_label.to_numpy())
    np.save('../../data/nid/y_test.npy', test_label.to_numpy())

if __name__ == '__main__':
    get_nid_dataset_raw(False)