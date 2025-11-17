#!/bin/bash

cd ..

python -m src.LPCA.lpca_with_sim_runner_new ZINC Dist 4 8 0.0 200
python -m src.LPCA.lpca_with_sim_runner_new Peptides Dist 4 8 0.0 200
python -m src.LPCA.lpca_with_sim_runner_new CIFAR Dist 4 12 0.0 200