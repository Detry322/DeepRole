#!/usr/bin/env bash

NS="$1"
NF="$2"
PC="$3"

cd code/nn_train

python train.py $NS $NF $PC

python convert.py models/${NS}_${NF}_${PC}.h5 exported_models/${NS}_${NF}_${PC}.json
