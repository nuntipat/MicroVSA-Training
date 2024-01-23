#!/bin/bash

echo "Preparing Qualcomm Keyword Speech Dataset..."
cd preprocessing/qksd && python prepare_dataset.py && cd ../../

echo "Preparing Free Spoken Digit Dataset..."
cd preprocessing/fsdd && python prepare_dataset.py && cd ../../

echo "Preparing WIreless Sensor Data Mining Dataset..."
cd preprocessing/wisdm && python generate_dataset.py && cd ../../

echo "Preparing ST multi-zone Time-of-Flight sensors hand posture recognition Dataset..."
cd preprocessing/st_handpose && python generate_dataset.py && cd ../../
