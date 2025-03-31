#!/bin/bash
echo "Starting AZH Runs"
echo "Starting Single Aug Runs"
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

echo "Starting Double Aug Runs"
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

echo "Starting Triple Aug Runs"
echo "Running with triple augmentation: geo_photo_elastic"
python ../train.py --triple_aug geo_photo_elastic --dataset azh_wound_dataset
echo "Running with triple augmentation: geo_photo_cutout"
python ../train.py --triple_aug geo_photo_cutout --dataset azh_wound_dataset
echo "Running with triple augmentation: geo_elastic_cutout"
python ../train.py --triple_aug geo_elastic_cutout --dataset azh_wound_dataset
echo "Running with triple augmentation: photo_elastic_cutout"
python ../train.py --triple_aug photo_elastic_cutout --dataset azh_wound_dataset

echo "Starting Quadro Aug Runs"
echo "Running with quadro augmentation: 'geo_photo_elastic_cutout'"
python ../train.py --quadro_aug 'geo_photo_elastic_cutout' --dataset azh_wound_dataset

echo "Starting Medtec Runs"
echo "Starting Single Aug Runs"
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

echo "Starting Double Aug Runs"
echo "Running with double augmentation: geo_photo"
python ../train.py --double_aug geo_photo --dataset medetec
echo "Running with double augmentation: geo_elastic"
python ../train.py --double_aug geo_elastic --dataset medetec
echo "Running with double augmentation: geo_cutout"
python ../train.py --double_aug geo_cutout --dataset medetec
echo "Running with double augmentation: photo_elastic"
python ../train.py --double_aug photo_elastic --dataset medetec
echo "Running with double augmentation: photo_cutout"
python ../train.py --double_aug photo_cutout --dataset medetec
echo "Running with double augmentation: elastic_cutout"
python ../train.py --double_aug elastic_cutout --dataset medetec

echo "Starting Triple Aug Runs"
echo "Running with triple augmentation: geo_photo_elastic"
python ../train.py --triple_aug geo_photo_elastic --dataset medetec
echo "Running with triple augmentation: geo_photo_cutout"
python ../train.py --triple_aug geo_photo_cutout --dataset medetec
echo "Running with triple augmentation: geo_elastic_cutout"
python ../train.py --triple_aug geo_elastic_cutout --dataset medetec
echo "Running with triple augmentation: photo_elastic_cutout"
python ../train.py --triple_aug photo_elastic_cutout --dataset medetec

echo "Starting Quadro Aug Runs"
echo "Running with quadro augmentation: 'geo_photo_elastic_cutout'"
python ../train.py --quadro_aug 'geo_photo_elastic_cutout' --dataset medetec






echo "All runs completed."
