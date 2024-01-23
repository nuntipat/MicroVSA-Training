# MicroVSA Model Training

This repository contain the model training and converting script for the binary LDC classifier and MCU-optimized LDC classifier described in the paper "MicroVSA: An Ultra-Lightweight Vector Symbolic Architecture-based Classifier Library for Always-On Inference on Tiny Microcontrollers".

## Prerequisite

- Python >=3.8
- virtualenv

## Dataset Preparation

1. PTB Diagnostic ECG Database *(A free Kagger's account is required to download this dataset)*
    1. Download the dataset from [link](https://www.kaggle.com/datasets/shayanfazeli/heartbeat/data)
    2. Unzip and copy the `ptbdb_normal.csv` and `ptbdb_abnormal.csv` to `data/ptb_ecg`

2. Qualcomm Keyword Speech Dataset *(A free Qualcomm developer network's account is required to download this dataset)*
    1. Download the dataset from [link](https://developer.qualcomm.com/project/keyword-speech-dataset)
    2. Unzip and copy the `qualcomm_keyword_speech_dataset` directory to the `data/qksd` directory

3. UCI Human Activity Recognition Dataset 
    1. Download the dataset from [link](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
    2. Unzip the file `UCI HAR Dataset.zip` and copy the text file (.txt) under `UCI HAR Dataset/train` and `UCI HAR Dataset/test` to `data/ucihar`

4. Free Spoken Digit Dataset
    1. Download the dataset from [link](https://github.com/Jakobovski/free-spoken-digit-dataset/releases/tag/v1.0.10)
    2. Unzip and copy the `free-spoken-digit-dataset-1.0.10` directory to the `data/fsdd` directory

5. WIreless Sensor Data Mining
    1. Download the dataset from [link](https://www.cis.fordham.edu/wisdm/dataset.php)
    2. Unzip and copy the `WISDM_ar_v1.1` directory to the `data/wisdm` directory 

6. ST multi-zone Time-of-Flight sensors hand posture recognition
    1. Download the dataset from [link](https://github.com/STMicroelectronics/stm32ai-modelzoo/raw/main/hand_posture/scripts/training/datasets/ST_VL53L8CX_handposture_dataset.zip)
    2. Unzip and copy the `ST_VL53L8CX_handposture_dataset` directory to the `data/st_handpose` directory

7. Microsoft Scalable Noisy Speech Dataset *(Use to generate negative sample to train the model on the Qualcomm Keyword Speech Dataset)*
    1. Download the dataset from [link](https://github.com/microsoft/MS-SNSD)
    2. Unzip and copy the dataset directory to the `data` directory

Final `data` directory structure

```
data/
    ptb_ecg/
        ptb_abnormal.csv
        ptb_normal.csv
    qksd/
        qualcomm_keyword_speech_dataset/
            hey_snapdragon/
            ...
    ucihar/
        X_train.txt
        y_train.txt
        X_test.txt
        y_test.txt
    fsdd/
        free-spoken-digit-dataset-1.0.10/
            acquire_data/
            recordings/
            ...
    widsm/
        WISDM_ar_v1.1/
    st_handpose/
        ST_VL53L8CX_handposture_dataset/
    MS-SNSD-master/
        clean_test/
        ...
```

## Usage

1. Create a Python virtual environment and setup all dependencies

    ```
    $ virtualenv -p python3 venv
    $ source venv/bin/activate
    $ ./setup.sh
    ```

2. Run the script to generate the training/testing dataset

    ```
    $ ./prepare_dataset.sh
    ```

3. Train the model

    Option 1) automatically train all models

    ```
    $ ./train.sh
    ```

    Option 2) train a model on a specific dataset with custom parameters e.g. the following command train five MCU-optimized LDC models (Df=64) on the UCI Human Activity Recognition Dataset. The training script saves all trained models in the `result` directory and report the best accuracy.

    ```bash
    # run `python train.py -h` to view all options
    $ python train.py -d har -n 5 -df 64
    ```

    The `result` directory contains the model in npy format, training log (log.txt) and image of the confusion matrix (result.png)

    ```
    data/
    preprocesssing/
    result/
        har_d64_1/
            C.npy
            F.npy
            V.npy
            log.txt
            result.png
        har_d64_2/
            ...
        ...
    ...
    ```

4. Convert the trained model to C source/header file for deploying on the MCU

    ```bash
    # run `python model_converter.py -h` to view all options
    $ python model_converter.py -i result/har_d64_1 -o result/har_d64_1 -n har -dv 8
    ```