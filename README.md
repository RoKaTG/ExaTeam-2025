# Monte Carlo Black-Scholes Pricing Optimization - Hackathon 2024

## Project Overview

This repository contains our solutions and optimizations for the Monte Carlo Black-Scholes Pricing problem & the code aster porting, as part of the Hackathon 2024 hosted by Viridien. The primary objective was to optimize the Black-Scholes pricing calculation through a Monte Carlo simulation on a Graviton 4 HPC cluster, leveraging advanced hardware and software techniques such as vectorization, OpenMP, MPI, and ACfL compiler-specific optimizations. Then using the said cluster to port a code with missing dependancies & prerequisities 

---

## Authors

- **Msilini Yassine**  
- **Sofiane Arhab**  
- **Matheo Pasquier**  
- **Ahmed Taleb Bechir**

---

## Repository Structure

### **Folder: `BSM/`**

This folder contains multiple versions of the Monte Carlo simulation code with incremental optimizations and architecture-specific enhancements. Below is a breakdown of the files:

| **File**             | **Description**                                                                                           |
|-----------------------|-----------------------------------------------------------------------------------------------------------|
| `BSM2.cxx`           | Initial, unoptimized version of the Monte Carlo Black-Scholes pricing implementation.                     |
| `BSM_SVE.cxx`        | Optimized version with SVE (Scalable Vector Extensions) for Graviton 4. Utilizes ACfL to leverage SVE.     |
| `BSM_assembly.cxx`   | Attempts inline assembly optimizations for critical sections, including RNG and payoff calculations.       |
| `BSM_fft.cxx`        | Implements a novel FFT-based acceleration for the Monte Carlo simulation (for path correlation).           |
| `BSM_final.cxx`      | The final, fully optimized version combining OpenMP, ACfL, and ArmPL routines.                       |
| `BSM_mpi.cxx`        | MPI-only parallel implementation, dividing simulations across multiple processes.                         |
| `BSM_open_mpi.cxx`   | Hybrid OpenMP + MPI implementation for scalable and multi-threaded distributed processing.                 |
| `BSM_openmp.cxx`     | OpenMP-optimized version for shared-memory parallelism on a single Graviton 4 node.                       |

### **Root Directory**

The root directory contains the necessary scripts to compile and benchmark the code:

| **File**              | **Description**                                                                                           |
|------------------------|-----------------------------------------------------------------------------------------------------------|
| `compile.sh`          | Script to compile all code versions in the `BSM/` folder. Handles compiler flags for ACfL and GCC.         |
| `benchmark.slurm`     | SLURM job script to benchmark all versions of the Monte Carlo Black-Scholes implementation on Graviton 4. |
| `final_bench.slurm`   | SLURM job script to benchmark **ACfL vs. GCC** on the **final version** (`BSM_final.cxx`). Includes runs to analyze **weak and strong scaling** on Graviton 4. |

---

## How to Use

### 1. **Compiling the Code**

To compile all versions of the code (the slurms script already execute the `benchmark.slurm` script), run:

```bash
./compile.sh
```

This script automatically detects the environment and compiles each file in the `BSM/` folder. By default, it uses ACfL with the following flags for Graviton 4:

```bash
armclang++ -Ofast -fopenmp -march=armv9-a+simd+fp16 -mcpu=neoverse-v2 -funroll-loops -ffast-math -fvectorize -larmpl -lamath -lm
```

For GCC, it uses equivalent architecture-specific flags.

### 2. **Benchmarking**

#### General Benchmarking (`benchmark.slurm`)

This script benchmarks all versions (`BSM2.cxx`, `BSM_SVE.cxx`, etc.) on the Graviton 4 cluster using SLURM. Submit the job with:

```bash
sbatch benchmark.slurm
```

The results will show runtime comparisons across all versions.

#### Final Version Benchmarking (`final_bench.slurm`)

This script compares the **final optimized version (`BSM_final.cxx`)** compiled with ACfL vs. GCC. It also performs **weak scalability** and **strong scalability** tests on the Graviton 4 cluster.

Submit the job with:

```bash
sbatch final_bench.slurm
```

Scalability results (e.g., scaling efficiency as threads/processes increase) are included in the output.

---

## Technical Highlights

- **Vectorization (SVE)**: Exploits Graviton 4's Scalable Vector Extensions for faster floating-point calculations.  
- **OpenMP and MPI**: Implemented for shared-memory and distributed-memory parallelism, optimizing for both single-node and multi-node scenarios.  
- **FFT Acceleration**: Introduced a fast Fourier transform (FFT)-based approach for correlated Monte Carlo paths.  
- **Inline Assembly**: Explored critical section optimization using ARM-specific assembly instructions.  
- **Arm Compiler for Linux (ACfL)**: Utilized for architecture-specific optimizations, significantly outperforming GCC in final benchmarks.  
- **SLURM Integration**: Automated job submission scripts for benchmarking and scalability analysis.  

---

## Performance Results

| **Compiler** | **Global Initial Seed** | **Argv[1]** | **Argv[2]** | **Value** | **Execution Time (s)** |
|--------------|--------------------------|-------------|-------------|-----------|-------------------------|
| **ACFL**     | 1107927401              | 100000      | 1000000     | 6.421719  | 9.646235               |
| **ACFL**     | 3607838896              | 1000000     | 1000000     | 6.421657  | 96.193266              |
| **ACFL**     | 3759633977              | 10000000    | 1000000     | 6.421636  | 961.795572             |
| **ACFL**     | 1587532951              | 100000000   | 1000000     | 6.421641  | 9620.493292            |
|              |                          |             |             |           |                         |
| **GCC**      | 763590619               | 100000      | 1000000     | 6.421719  | 27.027556              |
| **GCC**      | 3768789677              | 1000000     | 1000000     | 6.421657  | 270.040415             |
| **GCC**      | 1922258098              | 10000000    | 1000000     | 6.421636  | 2630.435618            |
| **GCC**      | 3945525756              | 100000000   | 1000000     | 6.421641  | 27043.345574           |

**Table: Benchmark Results Comparing ACFL vs GCC**
 

For detailed results, refer to our report.

---

## Authors' Notes

This project showcases how **domain-specific optimizations** (Monte Carlo simulations) can effectively leverage **hardware-specific tools** (ArmPL, ACfL) and **parallel paradigms** (OpenMP, MPI) to achieve significant speedups. It also highlights the importance of profiling tools (MAQAO) in guiding optimization decisions. 
