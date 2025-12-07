#!/bin/bash

cd ..

python -m src.LPCA.lpca_with_sim_runner ZINC SimPaths 4 8 1.0 # 200
# python -m src.LPCA.lpca_with_sim_runner Peptides SimPaths 4 36 0.3 # 200
# python -m src.LPCA.lpca_with_sim_runner CIFAR SimDegree 4 12 0.3 # 200