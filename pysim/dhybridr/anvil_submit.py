from pysim.parsing import File
from pysim.environment import dHybridRtemplate

main_submit_script: str = \
"""

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
"""

class AnvilSubmitScript(File):
    def __init__(
        self,
        path: str,
        queue: str = "wholenode",
        time_limit: str = "24:00:00",
        email: str|None = None,
        allocation: str = "phy220089"
    ) -> None:
        File.__init__(self, path, master=dHybridRtemplate.path+"/submit_anvil.sh")
        assert queue.lower() in ["wholenode", "debug"], f"Queue: {queue} not available, please choose either wholenode or debug"
        self.queue = queue.lower()
        self.time_limit = time_limit
        self.email = email
        self.allocation = allocation
        self.build()

    def __str__(self) -> str: return self.text
    def __repr__(self) -> str: return self.text

    def build(self) -> None: 
        self.header = "\n".join([
            "#!/bin/bash",
            f"#SBATCH -p {self.queue}      # Queue (partition) name",
            f"#SBATCH -t {self.time_limit}        # Run time (hh:mm:ss)",
            f"{'#' if self.email else '##'}SBATCH --mail-user={self.email}",
            f"{'#' if self.email else '##'}SBATCH --mail-type=all    # Send email at begin and end of job",
            f"#SBATCH -A {self.allocation}       # Allocation name (req'd if you have more than 1)"
        ])
        self.text = self.header + main_submit_script

    def write(self) -> None:
        with open(self.path, 'w') as file: file.write(self.text)
