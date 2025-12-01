#!/bin/bash

cd ..

python -m src.LPCA.lpca_with_sim_runner ZINC Sim 4 8 0.3 # 200
# python -m src.LPCA.lpca_with_sim_runner Peptides Sim 4 128 0.3 # 200
# python -m src.LPCA.lpca_with_sim_runner CIFAR Sim 4 64 0.3 # 200