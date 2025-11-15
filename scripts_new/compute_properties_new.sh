#!/bin/bash

cd ..

python -m src.compute.compute_properties ZINC
python -m src.compute.compute_properties Peptides
python -m src.compute.compute_properties CIFAR