#!/bin/bash
#SBATCH -p Prod
#SBATCH -A CSS
#SBATCH -n 200
#SBATCH -J ECV-BAMA-test01
#SBATCH --output=%J.out
#SBATCH --time=7-00:00:00
#SBATCH --mail-user=jfalgout@usgs.gov
#SBATCH --mail-type=ALL
#SBATCH -x compute[85,86,88,89]
#SBATCH -w compute87

module load openmpi-x86_64 epd-7.3-2-x86_64

INPUTDIR="/data/input"
OUTPUTDIR="/data/output/"
SHELFFILE="/data/landsat/FireECV/furry-ninja/python_mpi/East.shelf"
STACKFILE="test_tif_stack.csv"
#STACKFILE="tif_stack.csv"

#for PATHROW in `ls $INPUTDIR`; do
for PATHROW in p025r034; do
  mpiexec python mpi_StackProcessing.py -b ${INPUTDIR}/${PATHROW}/ -o ${OUTPUTDIR} -s ${SHELFFILE}  -sf ${STACKFILE} -t ${TMPDIR}/${PATHROW}/
done

exit
