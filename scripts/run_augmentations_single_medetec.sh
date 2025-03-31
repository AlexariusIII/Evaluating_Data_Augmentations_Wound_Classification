#!/bin/bash

echo "Running with augmentation: none"
python ../train.py --augmentation none --dataset medetec

echo "Running with augmentation: geometric"
python ../train.py --augmentation geometric --dataset medetec

echo "Running with augmentation: photometric"
python ../train.py --augmentation photometric --dataset medetec

echo "Running with augmentation: elastic"
python ../train.py --augmentation elastic --dataset medetec

echo "Running with augmentation: cutout"
python ../train.py --augmentation cutout --dataset medetec

echo "All runs completed."
