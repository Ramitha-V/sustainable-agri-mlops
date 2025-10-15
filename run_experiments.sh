#!/bin/bash

# Experiment Runner Script (Corrected)

echo "Starting experiments..."

# Experiment with number of estimators
for n_estimators in 50 100 150 200; do
  # Experiment with max depth
  for max_depth in 5 10 15 20 25; do
    echo "Running experiment with n_estimators=${n_estimators} and max_depth=${max_depth}"
    # Use --queue to run experiments in the background
    # CORRECTED: Use -S instead of -p to set parameters
    dvc exp run --queue -S params.yaml:train.n_estimators=${n_estimators} -S params.yaml:train.max_depth=${max_depth}
  done
done

echo "All experiments have been queued. DVC is running them now."