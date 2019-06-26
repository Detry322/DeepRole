#!/usr/bin/env bash

export HDF5_USE_FILE_LOCKING=FALSE

function train_section() {
    NUM_SUCCEEDS="$1"
    NUM_FAILS="$2"
    PROPOSE_COUNT="$3"

    echo "$(date) ======== Starting training process for: Succeeds=$NUM_SUCCEEDS, Fails=$NUM_FAILS, Propose=$PROPOSE_COUNT"
    echo "$(date) ==== Generating datapoints..."

    mkdir datapoint_output/${NUM_SUCCEEDS}_${NUM_FAILS}_${PROPOSE_COUNT}
    sbatch --wait --array=0-60 sbatch_generate_deeprole_data.sh $NUM_SUCCEEDS $NUM_FAILS $PROPOSE_COUNT

    DATAPOINT_COUNT=$(cat datapoint_output/${NUM_SUCCEEDS}_${NUM_FAILS}_${PROPOSE_COUNT}/* | wc -l)
    echo "$(date) ($NUM_SUCCEEDS, $NUM_FAILS, $PROPOSE_COUNT) Datapoints: $DATAPOINT_COUNT"

    # something like 
    echo "$(date) ==== Training neural network... (Takes 30-45 minutes)"
    sbatch --wait sbatch_train_neural_network.sh $NUM_SUCCEEDS $NUM_FAILS $PROPOSE_COUNT

    # This prints the last line of error (fragile)
    tail -40 $(ls sbatch_logs/* | grep train | grep .out | tail -1) | head -2

    echo "$(date) ==== Done with Succeeds=$NUM_SUCCEEDS, Fails=$NUM_FAILS, Propose=$PROPOSE_COUNT"
}

ITEMS="2 2
2 1
1 2
2 0
1 1
0 2
1 0
0 1
0 0"

IFS=$'\n'
for item in $ITEMS; do
    for i in $(seq 4 -1 0); do
        IFS=' ' read ns nf <<< "$item"
        train_section $ns $nf $i
    done
done
