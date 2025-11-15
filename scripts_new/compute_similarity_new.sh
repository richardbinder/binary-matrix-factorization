#!/bin/bash

cd ..

python -m src.compute.compute_encoding_similarity_new ZINC lpca_out/ lpca_with_sim_new_ZINC_methodSim_k8_b4_gamma0.3_s200
python -m src.compute.compute_encoding_similarity_new Peptides lpca_out/ lpca_with_sim_new_Peptides_methodSim_k8_b4_gamma0.3_s200
python -m src.compute.compute_encoding_similarity_new CIFAR lpca_out/ lpca_with_sim_new_CIFAR_methodSim_k12_b4_gamma0.3_s200
