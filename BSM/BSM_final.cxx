#include <cstdint>
#include <omp.h>
#include <sys/time.h>
#include <vector>
#include <random>
#include <iostream>
#include <iomanip>
#include <amath.h>
#include <armpl.h>

#define ui64 uint64_t

/*******************************************
 * @brief Returns the current time in microseconds.
 *
 * @return Current time in microseconds.
 *******************************************/
static double dml_micros() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return double(tv.tv_sec) * 1e6 + double(tv.tv_usec);
}

/*******************************************
 * @brief XORSHIFT128+ pseudo-random number generator initialization.
 *
 * @param st XORSHIFT state.
 * @param seed Initial seed.
 *******************************************/
struct xorshift128plus_state {
    ui64 s[2];
};

static inline void xorshift128plus_init(xorshift128plus_state &st, ui64 seed) {
    st.s[0] = seed;
    st.s[1] = seed ^ 0x9E3779B97F4A7C15ULL;
}

/*******************************************
 * @brief XORSHIFT128+ pseudo-random number generator.
 *
 * @param st XORSHIFT state.
 * @return Random 64-bit unsigned integer.
 *******************************************/
static inline ui64 xorshift128plus(xorshift128plus_state &st) {
    ui64 x = st.s[0];
    ui64 y = st.s[1];
    st.s[0] = y;
    x ^= x << 23;
    x ^= x >> 17;
    x ^= y ^ (y >> 26);
    st.s[1] = x;
    return x + y;
}

/*******************************************
 * @brief Box-Muller transform without rejection.
 *
 * @param u1 Uniform random variable in [0, 1).
 * @param u2 Uniform random variable in [0, 1).
 * @return Standard normal random variable.
 *******************************************/
__attribute__((always_inline)) static inline double box_muller_no_reject(double u1, double u2) {
    if (u1 < 1e-16) u1 = 1e-16; // Clamp to avoid log(0)
    double r = std::sqrt(-2.0 * std::log(u1));
    double theta = 2.0 * M_PI * u2;
    return r * std::cos(theta);
}

/*******************************************
 * @brief Approximates exponential with clamping.
 *
 * @param x Input value.
 * @return Approximated exponential value.
 *******************************************/
__attribute__((always_inline)) static inline double exp_approx_clamp(double x) {
    if (x < -10.0) x = -10.0;
    else if (x > 10.0) x = 10.0;
    double x2 = x * x;
    double x3 = x2 * x;
    return 1.0 + x + 0.5 * x2 + (1.0 / 6.0) * x3;
}

/*******************************************
 * @brief Monte Carlo kernel with fused approach.
 *
 * @param S0 Initial stock price.
 * @param K Strike price.
 * @param T Time to maturity.
 * @param r Risk-free interest rate.
 * @param sigma Volatility.
 * @param nSim Number of simulations.
 * @param runIndex Index of the current run.
 * @return Discounted mean payoff.
 *******************************************/
double black_scholes_monte_carlo_fused_noreject(
    double S0, double K, double T, double r, double sigma,
    ui64 nSim, ui64 runIndex) {
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol = sigma * std::sqrt(T);
    double disc = std::exp(-r * T);

    const int CHUNK = 256;
    ui64 nBlocks = nSim / CHUNK;
    ui64 reste = nSim % CHUNK;

    double payoffSum = 0.0;

    #pragma omp parallel reduction(+:payoffSum)
    {
        xorshift128plus_state rng;
        ui64 myThreadId = (ui64)omp_get_thread_num() + 1;
        ui64 seedBase = 0xDEADBEEF ^ (0xABCULL * myThreadId) ^ (0xA5ULL * (runIndex + 1));
        xorshift128plus_init(rng, seedBase);

        double* u1 = new double[CHUNK];
        double* u2 = new double[CHUNK];

        #pragma omp for schedule(static)
        for (ui64 b = 0; b < nBlocks; b++) {
            for (int i = 0; i < CHUNK; i++) {
                ui64 rA = xorshift128plus(rng);
                ui64 rB = xorshift128plus(rng);
                u1[i] = double(rA) * (1.0 / 18446744073709551616.0);
                u2[i] = double(rB) * (1.0 / 18446744073709551616.0);
            }

            double local = 0.0;
            #pragma omp simd reduction(+:local)
            for (int i = 0; i < CHUNK; i++) {
                double g = box_muller_no_reject(u1[i], u2[i]);
                double x = drift + vol * g;
                double e = exp_approx_clamp(x);
                double ST = S0 * e;
                double pay = (ST > K) ? (ST - K) : 0.0;
                local += pay;
            }
            payoffSum += local;
        }

        if (reste > 0) {
            for (ui64 i = 0; i < reste; i++) {
                ui64 rA = xorshift128plus(rng);
                ui64 rB = xorshift128plus(rng);
                u1[i] = double(rA) * (1.0 / 18446744073709551616.0);
                u2[i] = double(rB) * (1.0 / 18446744073709551616.0);
            }

            double local = 0.0;
            #pragma omp simd reduction(+:local)
            for (ui64 i = 0; i < reste; i++) {
                double g = box_muller_no_reject(u1[i], u2[i]);
                double x = drift + vol * g;
                double e = exp_approx_clamp(x);
                double ST = S0 * e;
                double pay = (ST > K) ? (ST - K) : 0.0;
                local += pay;
            }
            payoffSum += local;
        }

        delete[] u1;
        delete[] u2;
    }

    double meanPayoff = payoffSum / double(nSim);
    return disc * meanPayoff;
}

/*******************************************
 * @brief Main function for Monte Carlo Black-Scholes simulation.
 *
 * @param argc Argument count.
 * @param argv Argument values (num_sims and num_runs).
 * @return Execution status.
 *******************************************/
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_sims> <num_runs>\n";
        return 1;
    }

    ui64 nSim = std::stoull(argv[1]);
    ui64 nRuns = std::stoull(argv[2]);

    double S0 = 100.0;
    double K = 110.0;
    double T = 1.0;
    double r = 0.06;
    double sigma = 0.2;

    std::random_device rd;
    unsigned long long global_seed = rd();

    std::cout << "Global initial seed: " << global_seed
              << "   argv[1]= " << argv[1]
              << "   argv[2]= " << argv[2] << std::endl;

    double t1 = dml_micros();

    double sumVal = 0.0;
    #pragma omp parallel for reduction(+:sumVal)
    for (ui64 run = 0; run < nRuns; run++) {
        sumVal += black_scholes_monte_carlo_fused_noreject(
            S0, K, T, r, sigma, nSim, run
        );
    }

    double t2 = dml_micros();
    double meanVal = sumVal / double(nRuns);
    double elapsed = (t2 - t1) * 1e-6;

    std::cout << std::fixed << std::setprecision(6)
              << "value= " << meanVal
              << " in " << elapsed << " s\n";

    return 0;
}

