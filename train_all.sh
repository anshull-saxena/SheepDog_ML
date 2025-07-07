#!/bin/bash
 
for dataset in politifact gossipcop lun
do
    echo "Running SheepDog on dataset: $dataset"
    # Record start time in seconds
    start_time=$(date +%s)
    CUDA_VISIBLE_DEVICES=0 python src/sheepdog.py --dataset_name $dataset --model_name sheepdog --iters 3 --batch_size 4
    # Record end time and calculate elapsed time
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    # Convert to minutes and seconds for readability
    minutes=$((elapsed_time / 60))
    seconds=$((elapsed_time % 60))
    echo "Total time elapsed for $dataset: $minutes minutes and $seconds seconds"
done