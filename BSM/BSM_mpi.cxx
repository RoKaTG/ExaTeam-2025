#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <sys/time.h>
#include <mpi.h>

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
    static thread_local std::mt19937 generator(std::random_device{}());
    static thread_local std::normal_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
}

/*******************************************
 * @brief Approximates the square root using one iteration of Newton's method.
 *
 * @param x Input value.
 * @return Approximated square root.
 *******************************************/
inline double approx_sqrt(double x) {
    double guess = x * 0.5;
    guess = 0.5 * (guess + x / guess);
    return guess;
}

/*******************************************
 * @brief Approximates the exponential function using a truncated Taylor series.
 *
 * @param x Input value.
 * @return Approximated exponential value.
 *******************************************/
inline double approx_exp(double x) {
    if (x >  4.0) x =  4.0;
    if (x < -4.0) x = -4.0;
    double xx = x * x;
    double term1 = 1.0 + x;
    double term2 = 0.5 * xx;
    double term3 = (xx * x) / 6.0;
    return term1 + term2 + term3;
}

/*******************************************
 * @brief Computes the Black-Scholes call option price using the Monte Carlo method.
 *        This version uses loop unrolling, MPI for parallelization, and approximations.
 *
 * @param S0 Initial stock price.
 * @param K Strike price.
 * @param T Time to maturity.
 * @param r Risk-free interest rate.
 * @param sigma Volatility.
 * @param q Dividend yield.
 * @param local_num_sims Number of simulations for the local MPI process.
 * @return Discounted sum of payoffs from the simulations.
 *******************************************/
double black_scholes_monte_carlo_unroll_mpi_approx(
    ui64 S0, ui64 K, double T, double r, double sigma, double q,
    ui64 local_num_sims)
{
    double drift    = (r - q - 0.5 * sigma * sigma) * T;
    double vol      = sigma * approx_sqrt(T);
    double discount = approx_exp(-r * T);

    double sum_payoffs_local = 0.0;

    ui64 main_loop = (local_num_sims / 4) * 4;
    for (ui64 i = 0; i < main_loop; i += 4) {
        double Z0 = gaussian_box_muller();
        double Z1 = gaussian_box_muller();
        double Z2 = gaussian_box_muller();
        double Z3 = gaussian_box_muller();

        double ST0 = S0 * approx_exp(drift + vol * Z0);
        double ST1 = S0 * approx_exp(drift + vol * Z1);
        double ST2 = S0 * approx_exp(drift + vol * Z2);
        double ST3 = S0 * approx_exp(drift + vol * Z3);

        sum_payoffs_local += ((ST0 > K) ? (ST0 - K) : 0.0)
                           + ((ST1 > K) ? (ST1 - K) : 0.0)
                           + ((ST2 > K) ? (ST2 - K) : 0.0)
                           + ((ST3 > K) ? (ST3 - K) : 0.0);
    }
    for (ui64 i = main_loop; i < local_num_sims; i++) {
        double Z = gaussian_box_muller();
        double ST = S0 * approx_exp(drift + vol * Z);
        sum_payoffs_local += (ST > K) ? (ST - K) : 0.0;
    }

    return discount * sum_payoffs_local;
}

/*******************************************
 * @brief Main function to execute the Monte Carlo simulation with MPI.
 *
 * @param argc Argument count.
 * @param argv Argument values (num_simulations and num_runs).
 * @return Execution status.
 *******************************************/
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0)
            std::cerr << "Usage: mpirun -n <procs> " << argv[0]
                      << " <num_simulations> <num_runs>\n";
        MPI_Finalize();
        return 1;
    }

    ui64 num_simulations = std::stoull(argv[1]);
    ui64 num_runs        = std::stoull(argv[2]);

    ui64   S0     = 100;
    ui64   K      = 110;
    double T      = 1.0;
    double r      = 0.06;
    double sigma  = 0.2;
    double q      = 0.03;

    ui64 local_sims = num_simulations / size;
    ui64 remainder  = num_simulations % size;
    if ((ui64)rank < remainder) {
        local_sims++;
    }

    double t_start = 0.0;
    if (rank == 0) {
        std::random_device rd;
        unsigned long long global_seed = rd();
        std::cout << "Global initial seed: " << global_seed
                  << "   num_sims= " << num_simulations
                  << "   num_runs= " << num_runs
                  << "   size= " << size << std::endl;
        t_start = dml_micros();
    }

    double global_sum_value = 0.0;
    for (ui64 run = 0; run < num_runs; run++) {
        double local_value = black_scholes_monte_carlo_unroll_mpi_approx(
                                S0, K, T, r, sigma, q, local_sims);

        double total_payoffs = 0.0;
        MPI_Reduce(&local_value, &total_payoffs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double mean_run = total_payoffs / (double)num_simulations;
            global_sum_value += mean_run;
        }
    }

    if (rank == 0) {
        double t_end = dml_micros();
        double mean_value = global_sum_value / (double)num_runs;
        double elapsed_s  = (t_end - t_start) / 1e6;
        std::cout << std::fixed << std::setprecision(6)
                  << " value= " << mean_value
                  << " in "    << elapsed_s
                  << " seconds (Approx + MPI)\n";
    }

    MPI_Finalize();
    return 0;
}

