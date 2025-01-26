#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>

#define ui64 uint64_t

/*******************************************
 * @brief Returns the current time in microseconds.
 *
 * @return Current time in microseconds.
 *******************************************/
double dml_micros() {
    static struct timeval tv;
    static struct timezone tz;
    gettimeofday(&tv, &tz);
    return (tv.tv_sec * 1000000.0 + tv.tv_usec);
}

/*******************************************
 * @brief Generates Gaussian noise using the Box-Muller transform.
 *
 * @return Random Gaussian value.
 *******************************************/
double gaussian_box_muller() {
    static thread_local std::mt19937 gen(std::random_device{}());
    static thread_local std::normal_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}

/*******************************************
 * @brief Computes the Black-Scholes call option price using the Monte Carlo method.
 *        This version uses loop unrolling, MPI for parallelization, and OpenMP for threading.
 *
 * @param S0 Initial stock price.
 * @param K Strike price.
 * @param T Time to maturity.
 * @param r Risk-free interest rate.
 * @param sigma Volatility.
 * @param local_num_sims Number of simulations for the local MPI process.
 * @return Discounted sum of payoffs from the simulations.
 *******************************************/
double black_scholes_monte_carlo_hybrid_classic(
    double S0, double K, double T, double r, double sigma, ui64 local_num_sims)
{
    double drift    = (r - 0.5 * sigma * sigma) * T;
    double vol      = sigma * std::sqrt(T);
    double discount = std::exp(-r * T);

    double sum_payoffs_local = 0.0;

    ui64 main_loop = (local_num_sims / 4) * 4;

    #pragma omp parallel reduction(+:sum_payoffs_local)
    {
        #pragma omp for schedule(static)
        for (ui64 i = 0; i < main_loop; i += 4) {
            double Z0 = gaussian_box_muller();
            double Z1 = gaussian_box_muller();
            double Z2 = gaussian_box_muller();
            double Z3 = gaussian_box_muller();

            double ST0 = S0 * std::exp(drift + vol * Z0);
            double ST1 = S0 * std::exp(drift + vol * Z1);
            double ST2 = S0 * std::exp(drift + vol * Z2);
            double ST3 = S0 * std::exp(drift + vol * Z3);

            sum_payoffs_local += ((ST0 > K) ? (ST0 - K) : 0.0)
                               + ((ST1 > K) ? (ST1 - K) : 0.0)
                               + ((ST2 > K) ? (ST2 - K) : 0.0)
                               + ((ST3 > K) ? (ST3 - K) : 0.0);
        }

        #pragma omp for schedule(static)
        for (ui64 i = main_loop; i < local_num_sims; i++) {
            double Z = gaussian_box_muller();
            double ST = S0 * std::exp(drift + vol * Z);
            sum_payoffs_local += (ST > K) ? (ST - K) : 0.0;
        }
    }

    return discount * sum_payoffs_local;
}

/*******************************************
 * @brief Main function to execute the Monte Carlo simulation with MPI and OpenMP.
 *
 * @param argc Argument count.
 * @param argv Argument values (num_sims and num_runs).
 * @return Execution status.
 *******************************************/
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            std::cerr << "Usage: mpirun -n <procs> " 
                      << argv[0] << " <num_sims> <num_runs>\n";
        }
        MPI_Finalize();
        return 1;
    }

    ui64 num_sims = std::stoull(argv[1]);
    ui64 num_runs = std::stoull(argv[2]);

    double S0    = 100.0;
    double K     = 110.0;
    double T     = 1.0;
    double r     = 0.06;
    double sigma = 0.2;

    ui64 local_sims = num_sims / size;
    ui64 remainder  = num_sims % size;
    if ((ui64)rank < remainder) {
        local_sims++;
    }

    double t1 = 0.0;
    if (rank == 0) {
        std::random_device rd;
        unsigned long long global_seed = rd();
        std::cout << "Global initial seed: " << global_seed
                  << "  num_sims= " << num_sims
                  << "  num_runs= " << num_runs
                  << "  size= " << size << " (MPI)\n";
        #ifdef _OPENMP
        std::cout << "  Using OpenMP with max threads= " 
                  << omp_get_max_threads() << std::endl;
        #endif
        t1 = dml_micros();
    }

    double sum_total = 0.0;
    for (ui64 run = 0; run < num_runs; run++) {
        double local_val = black_scholes_monte_carlo_hybrid_classic(
                               S0, K, T, r, sigma, local_sims);

        double payoff_global = 0.0;
        MPI_Reduce(&local_val, &payoff_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double mean_run = payoff_global / (double)num_sims;
            sum_total += mean_run;
        }
    }

    if (rank == 0) {
        double t2 = dml_micros();
        double mean_value = sum_total / (double)num_runs;
        double elapsed_s  = (t2 - t1) / 1e6;
        std::cout << std::fixed << std::setprecision(6)
                  << " value= " << mean_value
                  << " in "    << elapsed_s
                  << " seconds (MPI+OpenMP hybrid)\n";
    }

    MPI_Finalize();
    return 0;
}
