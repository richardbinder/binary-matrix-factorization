#!/bin/bash

cd ..

# python -m src.LPCA.lpca_with_sim_runner_new ZINC 4 8 1.0 1000
python -m src.LPCA.lpca_with_sim_runner_new Peptides 4 8 1.0 50
# python -m src.LPCA.lpca_with_sim_runner_new CIFAR 4 12 1.0 50