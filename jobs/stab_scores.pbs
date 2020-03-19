#!/bin/bash
#PBS -N scores  # name of job
#PBS -l nodes=1:ppn=1 # how many nodes and processors per node
#PBS -l walltime=06:00:00
#PBS -l pmem=4gb  # how much RAM is needed
#PBS -A cyberlamp  # which allocation to use (either cyberlamp, open, or ebf11_a_g_sc_default)
#PBS -j oe  # put outputs and error info in the same file
#PBS -M cjg66@psu.edu  # set email
#PBS -m abe  # email me on abort, begin, or end of job
#PBS -t 0-1024%2000  #create a job array

echo "Starting job $PBS_JOBNAME"
date
starttime=$SECONDS
echo "Job id: $PBS_JOBID"
echo "Job arrayid: $PBS_ARRAYID"
echo "About to change into $PBS_O_WORKDIR"
cd $PBS_O_WORKDIR
echo "Running code"
python new_sys_stable.py $PBS_ARRAYID
date
echo "took $((SECONDS - starttime)) seconds"
echo "done :)"

# submit this this by going into this directory and writing "qsub HR858-noMCMC.pbs" into terminal