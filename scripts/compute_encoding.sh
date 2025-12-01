#!/bin/bash

cd ..

python -m src.LPCA.lpca_with_sim_runner ZINC SimDegree 4 8 0.3 # 200
python -m src.LPCA.lpca_with_sim_runner Peptides SimDegree 4 8 0.3 # 200
python -m src.LPCA.lpca_with_sim_runner CIFAR SimDegree 4 12 0.3 # 200