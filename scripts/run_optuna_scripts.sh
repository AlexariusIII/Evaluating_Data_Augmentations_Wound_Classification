#!/bin/bash
echo "Starting Optuna Runs"

echo "Starting Optuna Run Geometric"
python ../optuna_search_geometric.py --n_trials 50 --num_epochs 30

echo "Starting Optuna Run Photometric"
python ../optuna_search_photometric.py --n_trials 50 --num_epochs 30

echo "Starting Optuna Run Elastic"
python ../optuna_search_elastic.py --n_trials 50 --num_epochs 30

echo "Starting Optuna Run Cutout"
python ../optuna_search_cutout.py --n_trials 50 --num_epochs 30