#!/usr/bin/env bash

./train_end_to_end.sh 2>&1 | tee >(nc seashells.io 1337) end_to_end-$(date '+%Y-%m-%d_%H:%M:%S').log
