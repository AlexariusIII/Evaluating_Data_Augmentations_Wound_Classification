#!/bin/bash

echo "Running with double augmentation: geo_photo"
python ../train.py --double_aug geo_photo --dataset azh_wound_dataset

echo "Running with double augmentation: geo_elastic"
python ../train.py --double_aug geo_elastic --dataset azh_wound_dataset

echo "Running with double augmentation: geo_cutout"
python ../train.py --double_aug geo_cutout --dataset azh_wound_dataset

echo "Running with double augmentation: photo_elastic"
python ../train.py --double_aug photo_elastic --dataset azh_wound_dataset

echo "Running with double augmentation: photo_cutout"
python ../train.py --double_aug photo_cutout --dataset azh_wound_dataset

echo "Running with double augmentation: elastic_cutout"
python ../train.py --double_aug elastic_cutout --dataset azh_wound_dataset

echo "All runs completed."
