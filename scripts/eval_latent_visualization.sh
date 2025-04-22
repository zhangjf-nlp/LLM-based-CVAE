#!/bin/bash

datasets=("dailydialog" "agnews" "yelp")
vae_types=("FB-VAE", "DG-VAE" "AE" "Beta-VAE" "Beta02-VAE", "BOW-VAE")

for dataset_name in "${datasets[@]}"; do
    for vae_type in "${vae_types[@]}"; do
        args="--dataset_name ${dataset_name} --vae_type ${vae_type} --visualize_latent_and_position --small_test"
        
        if [ "$vae_type" = "DG-VAE" ]; then
            args="${args} --add_gskip_connection 1"
        fi
        
        # Execute the command
        python3 test_model_single_plus.py ${args}
    done
done