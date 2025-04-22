#!/bin/bash

datasets=("dailydialog")
vae_types=("DG-VAE" "AE" "Beta-VAE" "Beta02-VAE" "Optimus_")

for dataset_name in "${datasets[@]}"; do
    for vae_type in "${vae_types[@]}"; do
        args="--dataset_name ${dataset_name} --vae_type ${vae_type} --test_prior_inference"
        
        if [ "$vae_type" = "DG-VAE" ]; then
            args="${args} --add_gskip_connection 1"
        fi
        
        # Execute the command
        python3 test_model_single_plus.py ${args}
    done
done