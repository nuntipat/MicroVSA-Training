from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PowerTransformer, RobustScaler, QuantileTransformer
from tensorflow.keras.utils import to_categorical
import numpy as np

# adapt from https://github.com/fastmachinelearning/hls4ml-tutorial/blob/main/part1_getting_started.ipynb
def get_jsc_dataset():
    data = fetch_openml('hls4ml_lhc_jets_hlf')
    X, y = data['data'], data['target']

    le = LabelEncoder()
    y = le.fit_transform(y)

    transform = QuantileTransformer()
    X = transform.fit_transform(X)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    np.save('../../data/jsc/x_train.npy', X_train_val)
    np.save('../../data/jsc/x_test.npy', X_test)
    np.save('../../data/jsc/y_train.npy', y_train_val)
    np.save('../../data/jsc/y_test.npy', y_test)

if __name__ == '__main__':
    get_jsc_dataset()