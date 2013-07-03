#!/bin/bash
#SBATCH -p Prod
#SBATCH -A CSS
#SBATCH -n 420
#SBATCH -J ECV-BAMA-test01
#SBATCH --output=%J.out
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=jfalgout@usgs.gov
#SBATCH --mail-type=ALL

module load openmpi-x86_64 epd-7.3-2-x86_64

INPUTDIR="/data/input"

#for PATHROW in `ls $INPUTDIR`; do
for PATHROW in p025r034; do
  mpiexec python mpi_StackProcessing.py -b /data/input/$PATHROW/ -o /data/output/ -s ./East.shelf  -sf test_tif_stack.csv -t /work/$TMPDIR/$PATHROW/
done

exit
