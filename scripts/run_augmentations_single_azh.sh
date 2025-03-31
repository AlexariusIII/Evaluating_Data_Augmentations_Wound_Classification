#!/bin/bash

echo "Running with augmentation: none"
python ../train.py --augmentation none --dataset azh_wound_dataset

echo "Running with augmentation: geometric"
python ../train.py --augmentation geometric --dataset azh_wound_dataset

echo "Running with augmentation: photometric"
python ../train.py --augmentation photometric --dataset azh_wound_dataset

echo "Running with augmentation: elastic"
python ../train.py --augmentation elastic --dataset azh_wound_dataset

echo "Running with augmentation: cutout"
python ../train.py --augmentation cutout --dataset azh_wound_dataset

echo "All runs completed."
