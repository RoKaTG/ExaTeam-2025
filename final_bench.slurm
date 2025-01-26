#!/bin/bash
#SBATCH --job-name=run_BSM           
#SBATCH --output=output_%j.txt   
#SBATCH --error=error_%j.txt    
#SBATCH --time=200:00:00              
#SBATCH --ntasks=1                  
#SBATCH --cpus-per-task=96           
#SBATCH --mem=64G                   
#SBATCH --partition=c8g              

module use /tools/acfl/24.04/modulefiles
module load gnu
module load armpl/24.04.0
module use /tools/acfl/24.10/modulefiles/
module load acfl/24.10.1

export OMP_NUM_THREADS=96

./compil.sh

echo ""
echo "<=========== BENCH CLASSIQUE SET 1 ===========>"
echo ""
echo "CLANG"
./BSM_final 100000 1000000
echo "GCC"
./BSM_final_gcc 100000 1000000
echo ""

echo "<=========== BENCH CLASSIQUE SET 2 ===========>"
echo ""
echo "CLANG"
./BSM_final 1000000 1000000
echo "GCC"
./BSM_final_gcc 1000000 1000000
echo ""

echo "<=========== BENCH CLASSIQUE SET 3 ===========>"
echo ""
echo "CLANG"
./BSM_final 10000000 1000000
echo "GCC"
./BSM_final_gcc 10000000 1000000
echo ""

echo "<=========== BENCH CLASSIQUE SET 4 ===========>"
echo ""
echo "CLANG"
./BSM_final 100000000 1000000
echo "GCC"
./BSM_final_gcc 100000000 1000000
echo ""

# WEAK SCALING

echo ""
echo "<=========== WEAK SCALING 8 THREADS ===========>"
echo ""
export OMP_NUM_THREADS=8
echo "CLANG"
./BSM_final 100000 1000000
echo "GCC"
./BSM_final_gcc 100000 1000000
echo ""

echo "<=========== WEAK SCALING 12 THREADS ===========>"
echo ""
export OMP_NUM_THREADS=12
echo "CLANG"
./BSM_final 100000 1000000
echo "GCC"
./BSM_final_gcc 100000 1000000
echo ""

echo "<=========== WEAK SCALING 24 THREADS ===========>"
echo ""
export OMP_NUM_THREADS=24
echo "CLANG"
./BSM_final 100000 1000000
echo "GCC"
./BSM_final_gcc 100000 1000000
echo ""

echo "<=========== WEAK SCALING 48 THREADS ===========>"
echo ""
export OMP_NUM_THREADS=48
echo "CLANG"
./BSM_final 100000 1000000
echo "GCC"
./BSM_final_gcc 100000 1000000
echo ""

echo "<=========== WEAK SCALING 96 THREADS ===========>"
echo ""
export OMP_NUM_THREADS=96
echo "CLANG"
./BSM_final 100000 1000000
echo "GCC"
./BSM_final_gcc 100000 1000000
echo ""

# STRONG SCALING

echo ""
echo "<=========== STRONG SCALING SET 1 ===========>"
echo ""
export OMP_NUM_THREADS=6
echo "CLANG"
./BSM_final 1000 1000000
echo "GCC"
./BSM_final_gcc 1000 1000000
echo ""

echo "<=========== STRONG SCALING SET 2 ===========>"
echo ""
export OMP_NUM_THREADS=12
echo "CLANG"
./BSM_final 2000 1000000
echo "GCC"
./BSM_final_gcc 2000 1000000
echo ""

echo "<=========== STRONG SCALING SET 3 ===========>"
echo ""
export OMP_NUM_THREADS=24
echo "CLANG"
./BSM_final 4000 1000000
echo "GCC"
./BSM_final_gcc 4000 1000000
echo ""

echo "<=========== STRONG SCALING SET 4 ===========>"
echo ""
export OMP_NUM_THREADS=48
echo "CLANG"
./BSM_final 8000 1000000
echo "GCC"
./BSM_final_gcc 8000 1000000
echo ""

echo "<=========== STRONG SCALING SET 5 ===========>"
echo ""
export OMP_NUM_THREADS=96
echo "CLANG"
./BSM_final 16000 1000000
echo "GCC"
./BSM_final_gcc 16000 1000000
echo ""
 ""
echo "<=========== BENCHMARKS COMPLETED ===========>"
echo ""

rm BSM_final_gcc BSM_final
