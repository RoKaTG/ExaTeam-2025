#!/bin/bash

g++ -O BSM.cxx -o BSM
armclang++ -g3 -Ofast -fopenmp -march=armv9-a+simd+fp16 -mcpu=neoverse-v2 -funroll-loops -ffast-math -fvectorize -larmpl -lamath -lm BSM_openmp.cxx -o BSM_openmp
armclang++ -g3 -Ofast -fopenmp -march=armv9-a+simd+fp16 -mcpu=neoverse-v2 -funroll-loops -ffast-math -fvectorize -larmpl -lamath -lm BSM_mpi.cxx -o BSM_mpi
armclang++ -g3 -Ofast -fopenmp -march=armv9-a+simd+fp16 -mcpu=neoverse-v2 -funroll-loops -ffast-math -fvectorize -larmpl -lamath -lm BSM_open_mpi.cxx -o BSM_open_mpi
armclang++ -g3 -Ofast -fopenmp -march=armv9-a+simd+fp16 -mcpu=neoverse-v2 -funroll-loops -ffast-math -fvectorize -larmpl -lamath -lm BSM_fft.cxx -o BSM_fft
armclang++ -g3 -Ofast -fopenmp -march=armv9-a+simd+fp16 -mcpu=neoverse-v2 -funroll-loops -ffast-math -fvectorize -larmpl -lamath -lm BSM_SVE.cxx -o BSM_SVE
armclang++ -g3 -Ofast -fopenmp -march=armv9-a+simd+fp16 -mcpu=neoverse-v2 -funroll-loops -ffast-math -fvectorize -larmpl -lamath -lm BSM_assembly.cxx -o BSM_assembly
armclang++ -g3 -Ofast -fopenmp -march=armv9-a+simd+fp16 -mcpu=neoverse-v2 -funroll-loops -ffast-math -fvectorize -larmpl -lamath -lm BSM_final.cxx -o BSM_final
