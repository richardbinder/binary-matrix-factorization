#!/bin/bash

cd ..

# python -m src.compute.compute_encoding_similarity_new ZINC lpca_out/ lpca_with_sim_ZINC_k8_b4_gamma1.0_s1000
# python -m src.compute.compute_encoding_similarity_new Peptides lpca_out/ lpca_with_sim_Peptides_k8_b4_gamma1.0_s50
python -m src.compute.compute_encoding_similarity CIFAR lpca_out/ lpca_with_sim_CIFAR_k12_b4_gamma1.0_s50