#include <iostream>
#include <amath.h>
#include <armpl.h>
#include <cstdint>
#include <iomanip>
#include <omp.h>
#include <sys/time.h>

// ACLE SVE
#if !defined(__ARM_FEATURE_SVE)
#error "This code requires SVE intrinsics. Compile with -march=armv8.2-a+sve (or later)."
#endif

#include <arm_sve.h>

#define ui64 uint64_t

/*******************************************
 * @brief Returns the current time in microseconds.
 *
 * @return Current time in microseconds.
 *******************************************/
static double dml_micros() {
    struct timeval tv; gettimeofday(&tv, nullptr);
    return double(tv.tv_sec) * 1.0e6 + double(tv.tv_usec);
}

/*******************************************
 * @brief XORSHIFT128+ pseudo-random number generator initialization.
 *
 * @param st XORSHIFT state.
 * @param seed Initial seed.
 *******************************************/
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
 * @brief Approximation of exponential using SVE intrinsics.
 *
 * @param x Input vector.
 * @return Approximated exponential of the input vector.
 *******************************************/
static inline svfloat64_t sve_exp_approx_f64(svfloat64_t x) {
    svfloat64_t minv = svdup_f64(-10.0);
    svfloat64_t maxv = svdup_f64(10.0);
    svbool_t pg = svptrue_b64(); // Full predicate

    svfloat64_t xx = svmin_f64_m(pg, maxv, svmax_f64_m(pg, x, minv));

    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t x2 = svmul_f64_m(pg, xx, xx);
    svfloat64_t x3 = svmul_f64_m(pg, x2, xx);

    svfloat64_t c2 = svmul_f64_x(pg, x2, svdup_f64(0.5));
    svfloat64_t c3 = svmul_f64_x(pg, x3, svdup_f64(1.0 / 6.0));

    svfloat64_t s = svadd_f64_m(pg, one, xx);
    s = svadd_f64_m(pg, s, c2);
    s = svadd_f64_m(pg, s, c3);
    return s;
}

/*******************************************
 * @brief Monte Carlo kernel for Black-Scholes pricing using SVE.
 *
 * @param S0 Initial stock price.
 * @param K Strike price.
 * @param T Time to maturity.
 * @param r Risk-free rate.
 * @param sigma Volatility.
 * @param nSim Number of simulations.
 * @return Discounted payoff mean.
 *******************************************/
double black_scholes_monte_carlo_sve(
    double S0, double K, double T, double r, double sigma, ui64 nSim) {
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol = sigma * std::sqrt(T);
    double disc = std::exp(-r * T);

    double payoffSum = 0.0;

    #pragma omp parallel reduction(+:payoffSum)
    {
        xorshift128plus_state rng;
        ui64 seed = 0xABCDEF01ULL ^ (0x1234ULL * (omp_get_thread_num() + 1));
        xorshift128plus_init(rng, seed);

        #pragma omp for
        for (ui64 i = 0; i < nSim; i++) {
            ui64 ra = xorshift128plus(rng);
            ui64 rb = xorshift128plus(rng);
            double u1 = double(ra) * (1.0 / 18446744073709551616.0);
            double u2 = double(rb) * (1.0 / 18446744073709551616.0);

            double rBM = std::sqrt(-2.0 * std::log(u1));
            double theta = 2.0 * M_PI * u2;
            double g = rBM * std::cos(theta);

            double x = drift + vol * g;
            svfloat64_t vx = svdup_f64(x);
            svfloat64_t vex = sve_exp_approx_f64(vx);
            double ex0 = svlasta_f64(svptrue_b64(), vex);

            double ST = S0 * ex0;
            double pay = (ST > K) ? (ST - K) : 0.0;
            payoffSum += pay;
        }
    }

    double meanPay = payoffSum / double(nSim);
    return disc * meanPay;
}

/*******************************************
 * @brief Main function for Monte Carlo SVE Black-Scholes simulation.
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

    double S0 = 100.0, K = 110.0, T = 1.0, r = 0.06, sigma = 0.2;

    double t1 = dml_micros();
    double sumVal = 0.0;
    for (ui64 run = 0; run < nRuns; run++) {
        sumVal += black_scholes_monte_carlo_sve(S0, K, T, r, sigma, nSim);
    }
    double t2 = dml_micros();

    double meanVal = sumVal / double(nRuns);
    double elap = (t2 - t1) * 1e-6;
    std::cout << std::fixed << std::setprecision(6)
             << "SVE => value= " << meanVal
             << " in " << elap << " s\n";
    return 0;
}

