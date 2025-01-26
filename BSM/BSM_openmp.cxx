#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>  // OpenMP

#define ui64 uint64_t

/*******************************************
 * @brief Returns the current time in microseconds.
 * 
 * @return Current time in microseconds.
 *******************************************/
double dml_micros() {
    static struct timezone tz;
    static struct timeval  tv;
    gettimeofday(&tv, &tz);
    return (tv.tv_sec * 1000000.0) + tv.tv_usec;
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
 * @brief Computes the Black-Scholes call option price using the Monte Carlo method.
 *        This version uses loop unrolling and OpenMP for parallelization.
 *
 * @param S0 Initial stock price.
 * @param K Strike price.
 * @param T Time to maturity.
 * @param r Risk-free interest rate.
 * @param sigma Volatility.
 * @param q Dividend yield.
 * @param num_simulations Number of Monte Carlo simulations.
 * @return Call option price.
 *******************************************/
double black_scholes_monte_carlo_unroll_omp(
    ui64 S0, ui64 K, double T, double r, double sigma, double q, ui64 num_simulations)
{
    const double drift    = (r - q - 0.5 * sigma * sigma) * T; 
    const double vol      = sigma * std::sqrt(T);
    const double discount = std::exp(-r * T);
    const double invN     = 1.0 / static_cast<double>(num_simulations);

    double sum_payoffs = 0.0;

    #pragma omp parallel reduction(+:sum_payoffs)
    {
        #pragma omp for schedule(static)
        for (ui64 i = 0; i < (num_simulations / 4) * 4; i += 4) {
            double Z0 = gaussian_box_muller();
            double Z1 = gaussian_box_muller();
            double Z2 = gaussian_box_muller();
            double Z3 = gaussian_box_muller();

            double ST0 = S0 * std::exp(drift + vol * Z0);
            double ST1 = S0 * std::exp(drift + vol * Z1);
            double ST2 = S0 * std::exp(drift + vol * Z2);
            double ST3 = S0 * std::exp(drift + vol * Z3);

            double payoff0 = (ST0 > K) ? (ST0 - K) : 0.0;
            double payoff1 = (ST1 > K) ? (ST1 - K) : 0.0;
            double payoff2 = (ST2 > K) ? (ST2 - K) : 0.0;
            double payoff3 = (ST3 > K) ? (ST3 - K) : 0.0;

            sum_payoffs += (payoff0 + payoff1 + payoff2 + payoff3);
        }

        #pragma omp for schedule(static)
        for (ui64 i = (num_simulations / 4) * 4; i < num_simulations; i++) {
            double Z = gaussian_box_muller();
            double ST = S0 * std::exp(drift + vol * Z);
            double payoff = (ST > K) ? (ST - K) : 0.0;
            sum_payoffs += payoff;
        }
    }

    return discount * (sum_payoffs * invN);
}

/*******************************************
 * @brief Main function to execute the Monte Carlo simulation.
 *
 * @param argc Argument count.
 * @param argv Argument values (num_simulations and num_runs).
 * @return Execution status.
 *******************************************/
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_simulations> <num_runs>\n";
        return 1;
    }

    ui64 num_simulations = std::stoull(argv[1]);
    ui64 num_runs        = std::stoull(argv[2]);

    // Parameters
    ui64   S0     = 100;
    ui64   K      = 110;
    double T      = 1.0;
    double r      = 0.06;
    double sigma  = 0.2;
    double q      = 0.03;

    std::random_device rd;
    unsigned long long global_seed = rd();
    std::cout << "Global initial seed: " << global_seed
              << "  argv[1]= " << argv[1]
              << "  argv[2]= " << argv[2] << std::endl;

    double t1 = dml_micros();
    double sum = 0.0;
    for (ui64 run = 0; run < num_runs; ++run) {
        sum += black_scholes_monte_carlo_unroll_omp(S0, K, T, r, sigma, q, num_simulations);
    }
    double t2 = dml_micros();

    double mean_value = sum / static_cast<double>(num_runs);
    double elapsed_s  = (t2 - t1) / 1e6;

    std::cout << std::fixed << std::setprecision(6)
              << " value= " << mean_value
              << " in "    << elapsed_s << " seconds (NoApprox + OMP)\n";

    return 0;
}

