#!/bin/bash

# Train the model
echo "Training the model..."
python main.py --mode train --train_data churn-bigml-80.csv --test_data churn-bigml-20.csv --save model.pkl

# Evaluate the model
echo "Evaluating the model..."
python main.py --mode evaluate --load model.pkl --test_data churn-bigml-20.csv

# Load the model (just to verify loading works)
echo "Loading the model..."
python main.py --mode load --load model.pkl

