#!/bin/bash

echo "PTB Diagnostic ECG Database (w/o sgn, Df=32)"
python train.py -d ptb -n $1 -df 32
echo "PTB Diagnostic ECG Database (w/ sgn, Df=256)"
python train.py -d ptb -n $1 --use-sgn -df 256
echo

echo "Qualcomm Keyword Speech Dataset (w/o sgn, Df=32)"
python train.py -d qksd -n $1 -df 32
echo "Qualcomm Keyword Speech Dataset (w/ sgn, Df=256)"
python train.py -d qksd -n $1 --use-sgn -df 256
echo

echo "UCI Human Activity Recognition Dataset (w/o sgn, Df=128)"
python train.py -d har -n $1 -df 128
echo "UCI Human Activity Recognition Dataset (w/ sgn, Df=256)"
python train.py -d har -n $1 --use-sgn -df 256
echo

echo "Free Spoken Digit Dataset (w/o sgn, Df=64)"
python train.py -d fsdd -n $1 -df 64
echo "Free Spoken Digit Dataset (w/ sgn, Df=256)"
python train.py -d fsdd -n $1 --use-sgn -df 256
echo

echo "WIreless Sensor Data Mining (w/o sgn, Df=64)"
python train.py -d wisdm -n $1 -df 64
echo

echo "ST multi-zone Time-of-Flight sensors hand posture recognition (w/ sgn, Df=192)"
python train.py -d sthand -n $1 --use-sgn -df 192
echo