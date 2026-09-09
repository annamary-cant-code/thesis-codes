#!/bin/bash
#SBATCH --job-name=inclinata
#SBATCH --partition=compute
#SBATCH --account=education-ae-msc-ae
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=slurm-%j.out

module load ls-dyna/13.0.1
module load 13.0.1/smp_d
ls-dyna_smp_d_R13_1_0_x64_centos79_ifort190 i=RUN_inclinata.k memory=260m ncpu=48
