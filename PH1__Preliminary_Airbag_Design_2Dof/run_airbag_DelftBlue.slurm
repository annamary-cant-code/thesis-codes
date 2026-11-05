#!/bin/bash
#SBATCH --job-name=airbag_DE
#SBATCH --partition=compute
#SBATCH --account=education-ae-msc-ae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=1G
#SBATCH --time=20:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm-%j.out

cd $SLURM_SUBMIT_DIR

module load 2024r1
module load python
source $HOME/venv_airbag/bin/activate

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RUN_SIZE=${RUN_SIZE:-long}
LOG=airbag_DE_${RUN_SIZE}_${SLURM_JOB_ID}.LOG
echo "=== job $SLURM_JOB_ID on $(hostname), cpus=$SLURM_CPUS_PER_TASK, $(date) ===" >> $LOG
python -u differential_evolution_wrapper.py $RUN_SIZE >> $LOG 2>&1
echo "=== exit code $?, $(date) ===" >> $LOG