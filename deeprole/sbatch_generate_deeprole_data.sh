#!/bin/bash
#SBATCH --job-name=generate_deeprole_data
#SBATCH --output=sbatch_logs/generate_deeprole-%j.out
#SBATCH --error=sbatch_logs/generate_deeprole-%j.err
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=jserrino
#SBATCH --requeue

NS="$1"
NF="$2"
PC="$3"

PREFIX=$(openssl rand -hex 4)

for i in $(seq 1 8); do
    sleep 0.2
    ./deeprole -n250 -i1500 -w500 -p$PC -s$NS -f$NF --out=datapoint_output/${NS}_${NF}_${PC} --modeldir=deeprole_models --suffix=$PREFIX-$i &
    pids[$i]=$!
done

# wait for all pids
for pid in ${pids[*]}; do
    wait $pid
done
