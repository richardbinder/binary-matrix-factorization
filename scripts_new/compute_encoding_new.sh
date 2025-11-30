#!/bin/bash

cd ..

python -m src.LPCA.lpca_with_sim_runner ZINC 4 8 0.0
# python -m src.LPCA.lpca_with_sim_runner Peptides 4 8 0.0 50
# python -m src.LPCA.lpca_with_sim_runner CIFAR 4 12 2.0 50