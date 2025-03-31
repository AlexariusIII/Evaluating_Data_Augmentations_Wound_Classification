#!/bin/bash

echo "Running with triple augmentation: geo_photo_elastic"
python ../train.py --triple_aug geo_photo_elastic --dataset azh_wound_dataset

echo "Running with triple augmentation: geo_photo_cutout"
python ../train.py --triple_aug geo_photo_cutout --dataset azh_wound_dataset

echo "Running with triple augmentation: geo_elastic_cutout"
python ../train.py --triple_aug geo_elastic_cutout --dataset azh_wound_dataset

echo "Running with triple augmentation: photo_elastic_cutout"
python ../train.py --triple_aug photo_elastic_cutout --dataset azh_wound_dataset

echo "All runs completed."
