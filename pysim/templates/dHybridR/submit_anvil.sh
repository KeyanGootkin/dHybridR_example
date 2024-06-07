#!/bin/bash
#SBATCH -p wholenode      # Queue (partition) name
#SBATCH -t 24:00:00        # Run time (hh:mm:ss)
##SBATCH --mail-user=keyangootkin@gmail.com
##SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A phy220089       # Allocation name (req'd if you have more than 1)

module load intel
module load hdf5

if [ -z ${SLURM_JOB_ID+x} ]; then
    n1=`awk '/node_number/{split($1, a, "="); split(a[2], b, ","); print b[1]}' input/input`
    n2=`awk '/node_number/{split($1, a, "="); split(a[2], b, ","); print b[2]}' input/input`
    n3=`awk '/node_number/{split($1, a, "="); split(a[2], b, ","); print b[3]}' input/input`
    n1=${n1:=1}
    n2=${n2:=1}
    n3=${n3:=1}
    ntasks=`echo $n1*$n2*$n3 | bc`
    CDN=${PWD##*/}
    taskspernode=128
    nodes=`echo $ntasks/$taskspernode | bc`
    remainder=`echo $ntasks%$taskspernode | bc`
    if (( remainder > 0 )); then
        nodes=`echo $nodes+1 | bc`
    fi
    echo x_tasks=$n1, y_tasks=$n2, z_tasks=$n3, n_tasks=$ntasks, nodes=$nodes
    sbatch --ntasks=$ntasks --nodes=$nodes --job-name=$CDN --output=$CDN.out --error=$CDN.err $0

else
    echo $SLURM_JOB_ID
    mpirun dHybridR > out
fi
