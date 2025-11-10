#!/bin/bash

cd ..

# python -m src.compute.compute_encoding_similarity ZINC lpca_out/ lpca_with_sim_ZINC_k8_b4_gamma0.0_s1000
# python -m src.compute.compute_encoding_similarity Peptides lpca_out/ lpca_with_sim_Peptides_k8_b4_gamma0.0_s50
python -m src.compute.compute_encoding_similarity CIFAR lpca_out/ lpca_with_sim_CIFAR_k32_b4_gamma2.0_s50