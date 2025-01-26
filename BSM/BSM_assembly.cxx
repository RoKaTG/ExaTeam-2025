#include <iostream>
#include <amath.h>
#include <armpl.h>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>

#define ui64 uint64_t

/*******************************************
 * @brief Returns the current time in microseconds.
 *
 * Used for performance measurement.
 * @return Current time in microseconds.
 *******************************************/
static double dml_micros(){
    struct timeval tv; 
    gettimeofday(&tv, nullptr);
    return double(tv.tv_sec) * 1.0e6 + tv.tv_usec;
}

/*******************************************
 * @brief XORSHIFT128+ RNG state structure.
 *
 * Holds the state for the random number generator.
 *******************************************/
struct xorshift128plus_state {
    ui64 s[2]; // Two 64-bit integers for RNG state.
};

/*******************************************
 * @brief Initializes the XORSHIFT128+ RNG state.
 *
 * @param st The RNG state to initialize.
 * @param seed The seed for random number generation.
 *******************************************/
static inline void xorshift128plus_init(xorshift128plus_state &st, ui64 seed){
    st.s[0] = seed;
    st.s[1] = seed ^ 0x9E3779B97F4A7C15ULL; // XOR with a constant for variation.
}

/*******************************************
 * @brief Generates a random 64-bit unsigned integer.
 *
 * @param st The RNG state.
 * @return A random 64-bit integer.
 *******************************************/
static inline ui64 xorshift128plus(xorshift128plus_state &st){
    ui64 x = st.s[0];
    ui64 y = st.s[1];
    st.s[0] = y;
    x ^= x << 23; // Bitwise XOR and shifts for randomization.
    x ^= x >> 17;
    x ^= y ^ (y >> 26);
    st.s[1] = x;
    return x + y; // Combines the two numbers for the final random value.
}

/*******************************************
 * @brief Generates two Gaussian random variables using Box-Muller transform.
 *
 * @param u1 First uniform random variable in [0, 1).
 * @param u2 Second uniform random variable in [0, 1).
 * @param g1 First Gaussian random variable (output).
 * @param g2 Second Gaussian random variable (output).
 *******************************************/
static inline void approx_box_muller_2(double u1, double u2, double &g1, double &g2){
    if (u1 < 1e-16) u1 = 1e-16; // Avoid log(0).
    if (u1 > 1.0 - 1e-16) u1 = 1.0 - 1e-16; // Clamp values near 1.
    double r = std::sqrt(-2.0 * std::log(u1)); // Radius computation.
    double theta = 2.0 * M_PI * u2; // Angle computation.
    g1 = r * std::cos(theta); // First Gaussian variable.
    g2 = r * std::sin(theta); // Second Gaussian variable.
}

/*******************************************
 * @brief Approximates the exponential function using inline assembly.
 *
 * Implements the approximation e^x ≈ 1 + x + x^2/2 + x^3/6.
 * @param x The input value.
 * @return Approximated exponential value.
 *******************************************/
static inline double sve_exp_asm(double x) {
    double res; // Result variable.
    asm volatile(
        " fmov d0, %1        \n" // Move input x into register d0.
        " fmul d1, d0, d0    \n" // Compute x^2 in d1.
        " fmul d2, d1, d0    \n" // Compute x^3 in d2.
        " fadd d3, %2, d0    \n" // d3 = c1 + x.
        " fmadd d3, d1, %3, d3 \n" // d3 += x^2 * c05.
        " fmadd d3, d2, %4, d3 \n" // d3 += x^3 * c166.
        " fmov %0, d3        \n" // Move result from d3 to res.
        : "=w"(res) // Output operand.
        : "w"(x), "w"(c1), "w"(c05), "w"(c166) // Input operands.
        : "d0", "d1", "d2", "d3", "memory", "cc" // Clobbered registers.
    );
    return res;
}

/*******************************************
 * @brief Monte Carlo kernel optimized with inline assembly for SVE.
 *
 * @param S0 Initial stock price.
 * @param K Strike price.
 * @param T Time to maturity.
 * @param r Risk-free interest rate.
 * @param sigma Volatility.
 * @param nSim Number of simulations.
 * @return Discounted mean payoff.
 *******************************************/
double black_scholes_monte_carlo_optimized(
    double S0, double K, double T, double r, double sigma, ui64 nSim) {
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol = sigma * std::sqrt(T);
    double disc = std::exp(-r * T);

    double payoffSum = 0.0;

    #pragma omp parallel reduction(+:payoffSum)
    {
        xorshift128plus_state rng;
        ui64 seedBase = 0xDEADBEEF ^ (0x12345678ULL * (omp_get_thread_num() + 1));
        xorshift128plus_init(rng, seedBase);

        #pragma omp for simd schedule(static)
        for (ui64 i = 0; i < (nSim / 4) * 4; i += 4) {
            // Generate 4 random numbers.
            ui64 r1 = xorshift128plus(rng);
            ui64 r2 = xorshift128plus(rng);
            ui64 r3 = xorshift128plus(rng);
            ui64 r4 = xorshift128plus(rng);

            // Convert to uniform [0, 1).
            double u1 = double(r1) * (1.0 / 18446744073709551616.0);
            double u2 = double(r2) * (1.0 / 18446744073709551616.0);
            double u3 = double(r3) * (1.0 / 18446744073709551616.0);
            double u4 = double(r4) * (1.0 / 18446744073709551616.0);

            // Generate Gaussian random variables.
            double g1, g2, g3, g4;
            approx_box_muller_2(u1, u2, g1, g2);
            approx_box_muller_2(u3, u4, g3, g4);

            // Calculate drift and volatility.
            double x1 = drift + vol * g1;
            double x2 = drift + vol * g2;
            double x3 = drift + vol * g3;
            double x4 = drift + vol * g4;

            // Approximate exponential values.
            double e1 = sve_exp_asm(x1);
            double e2 = sve_exp_asm(x2);
            double e3 = sve_exp_asm(x3);
            double e4 = sve_exp_asm(x4);

            // Compute terminal stock prices.
            double ST1 = S0 * e1;
            double ST2 = S0 * e2;
            double ST3 = S0 * e3;
            double ST4 = S0 * e4;

            // Calculate payoffs.
            double p1 = (ST1 > K) ? (ST1 - K) : 0.0;
            double p2 = (ST2 > K) ? (ST2 - K) : 0.0;
            double p3 = (ST3 > K) ? (ST3 - K) : 0.0;
            double p4 = (ST4 > K) ? (ST4 - K) : 0.0;

            // Accumulate the payoffs.
            payoffSum += (p1 + p2 + p3 + p4);
        }

        // Handle remaining simulations.
        #pragma omp for simd schedule(static)
        for (ui64 i = (nSim / 4) * 4; i < nSim; i++) {
            ui64 r1 = xorshift128plus(rng);
            ui64 r2 = xorshift128plus(rng);
            double u1 = double(r1) * (1.0 / 18446744073709551616.0);
            double u2 = double(r2) * (1.0 / 18446744073709551616.0);

            double g1, g2;
            approx_box_muller_2(u1, u2, g1, g2);

            double x = drift + vol * g1;
            double e = sve_exp_asm(x);
            double ST = S0 * e;
            double pay = (ST > K) ? (ST - K) : 0.0;
            payoffSum += pay;
        }
    }

    return disc * (payoffSum / double(nSim));
}

/*******************************************
 * @brief Main function for the simulation.
 *
 * Parses input arguments and runs the simulation.
 * @param argc Argument count.
 * @param argv Argument values.
 * @return Execution status.
 *******************************************/
int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_sims> <num_runs>\n";
        return 1;
    }

    ui64 nSim = std::stoull(argv[1]);
    ui64 nRuns = std::stoull(argv[2]);

    double S0 = 100.0, K = 110.0, T = 1.0, r = 0.06, sigma = 0.2;

    double t1 = dml_micros();

    double sumVal = 0.0;
    #pragma omp parallel for reduction(+:sumVal)
    for (ui64 run = 0; run < nRuns; run++) {
        sumVal += black_scholes_monte_carlo_optimized(S0, K, T, r, sigma, nSim);
    }

    double t2 = dml_micros();

    double meanVal = sumVal / double(nRuns);
    double elapsed = (t2 - t1) * 1e-6;

    std::cout << std::fixed << std::setprecision(6)
              << "SVE ASM => value= " << meanVal
              << " in " << elapsed << " s\n";
    return 0;
}

