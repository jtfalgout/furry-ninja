#!/bin/bash
#SBATCH -p Prod
#SBATCH -A Ecosystems
#SBATCH -n 160
#SBATCH -J ECV-BAMA-full01
#SBATCH --output=%J.out
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=jfalgout@usgs.gov
#SBATCH --mail-type=ALL
#SBATCH -x compute[85,86,88,89]
##SBATCH -w compute[87,188-192]
#SBATCH --mincpus=16
#SBATCH --ntasks-per-node=16

module load openmpi-x86_64 epd-7.3-2-x86_64

INPUTDIR="/data/input"
OUTPUTDIR="/data/output/"
SHELFFILE="/data/landsat/FireECV/furry-ninja/python_mpi/East.shelf"
STACKFILE="tif_stack.csv"
#STACKFILE="test_tif_stack.csv"

#for PATHROW in p025r034; do
for PATHROW in `ls $INPUTDIR`; do
  mpiexec python mpi_StackProcessing.py -b ${INPUTDIR}/${PATHROW}/ -o ${OUTPUTDIR} -s ${SHELFFILE}  -sf ${STACKFILE} -t ${TMPDIR}/${PATHROW}/
done

exit
