#!/bin/bash
#SBATCH --job-name=run_BSM           
#SBATCH --output=output_BSM_%j.txt   
#SBATCH --error=error_BSM_%j.txt    
#SBATCH --time=05:00:00              
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=96           
#SBATCH --mem=64G                   
#SBATCH --partition=c8g              

module use /tools/acfl/24.04/modulefiles
module load gnu
module load armpl/24.04.0
module load acfl/24.10.1

export OMP_NUM_THREADS=96

./compile.sh

echo "<------------------------------Run with 96 threads------------------------------->"
./BSM 10000 100000
./BSM_openmp 10000 100000
./BSM_mpi 10000 100000
./BSM_open_mpi 10000 100000
./BSM_fft 10000 100000
./BSM_SVE 10000 100000
./BSM_assembly 10000 100000
./BSM_final 10000 100000

rm BSM BSM_openmp BSM_open_mpi BSM_mpi BSM_fft BSM_SVE BSM_assembly BSM_final
