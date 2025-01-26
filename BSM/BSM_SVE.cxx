#include <iostream>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <sys/time.h>
#include <vector>
#include <arm_sve.h>

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
    return double(tv.tv_sec) * 1e6 + tv.tv_usec;
}

/*******************************************
 * @brief XORSHIFT128+ pseudo-random number generator state.
 *
 * Holds the state for the RNG.
 *******************************************/
struct xorshift128plus_state {
    ui64 s[2]; // Internal state of the generator.
};

/*******************************************
 * @brief Initializes the XORSHIFT128+ RNG state.
 *
 * @param st The state to initialize.
 * @param seed The seed value for initialization.
 *******************************************/
static inline void xorshift128plus_init(xorshift128plus_state &st, ui64 seed){
    st.s[0] = seed;
    st.s[1] = seed ^ 0x9E3779B97F4A7C15ULL; // Combines seed with a constant.
}

/*******************************************
 * @brief Generates a random 64-bit unsigned integer.
 *
 * @param st The state of the RNG.
 * @return A 64-bit random number.
 *******************************************/
static inline ui64 xorshift128plus(xorshift128plus_state &st){
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
 * @brief Generates two Gaussian random variables using the Box-Muller transform.
 *
 * @param u1 First uniform random variable in [0, 1).
 * @param u2 Second uniform random variable in [0, 1).
 * @param g1 Output Gaussian random variable.
 * @param g2 Output Gaussian random variable.
 *******************************************/
static inline void box_muller_2(double u1, double u2, double &g1, double &g2){
    if(u1 < 1e-16) u1 = 1e-16;  // Avoid log(0).
    if(u1 > 1.0 - 1e-16) u1 = 1.0 - 1e-16;
    double r = std::sqrt(-2.0 * std::log(u1));
    double theta = 2.0 * M_PI * u2;
    g1 = r * std::cos(theta);
    g2 = r * std::sin(theta);
}

/*******************************************
 * @brief Approximates the exponential function using SVE intrinsics.
 *
 * The approximation uses a truncated Taylor series.
 * @param x Input vector of values.
 * @return Approximated exponential values for the input vector.
 *******************************************/
static inline svfloat64_t sve_exp_approx(svfloat64_t x)
{
    svbool_t pg = svptrue_b64(); // Active predicate for all lanes.
    svfloat64_t mn = svdup_f64(-10.0); // Minimum clamp value.
    svfloat64_t mx = svdup_f64(10.0);  // Maximum clamp value.
    x = svmax_f64_m(pg, x, mn); // Clamp to minimum.
    x = svmin_f64_m(pg, x, mx); // Clamp to maximum.

    // Polynomial approximation: e^x ~ 1 + x + x^2/2 + x^3/6
    svfloat64_t one = svdup_f64(1.0);
    svfloat64_t x2 = svmul_f64_m(pg, x, x); // x^2
    svfloat64_t x3 = svmul_f64_m(pg, x2, x); // x^3

    svfloat64_t c2 = svmul_f64_m(pg, x2, svdup_f64(0.5));
    svfloat64_t c3 = svmul_f64_m(pg, x3, svdup_f64(1.0 / 6.0));

    svfloat64_t s = svadd_f64_m(pg, one, x);
    s = svadd_f64_m(pg, s, c2);
    s = svadd_f64_m(pg, s, c3);
    return s;
}

/*******************************************
 * @brief Calculates the payoff for a European call option.
 *
 * The payoff is max(ST - K, 0).
 * @param st Vector of terminal stock prices.
 * @param K Strike price.
 * @param pg Predicate for active lanes.
 * @return Vector of payoffs.
 *******************************************/
static inline svfloat64_t sve_payoff(
    svfloat64_t st, svfloat64_t K, svbool_t pg)
{
    svfloat64_t diff = svsub_f64_m(pg, st, K); // ST - K
    svbool_t mpos = svcmpgt_f64(pg, diff, svdup_f64(0.0)); // diff > 0
    svfloat64_t pay = svsel_f64(mpos, diff, svdup_f64(0.0)); // Select positive values.
    return pay;
}

/*******************************************
 * @brief Monte Carlo kernel for Black-Scholes pricing using SVE.
 *
 * Uses vectorized operations for performance.
 * @param S0 Initial stock price.
 * @param K Strike price.
 * @param T Time to maturity.
 * @param r Risk-free rate.
 * @param sigma Volatility.
 * @param nSim Number of simulations.
 * @return Discounted mean payoff.
 *******************************************/
double black_scholes_monte_carlo_svechunk(
    double S0, double K, double T, double r, double sigma, ui64 nSim)
{
    double drift = (r - 0.5 * sigma * sigma) * T;
    double vol = sigma * std::sqrt(T);
    double disc = std::exp(-r * T);

    const int CHUNK = 512; // Number of random samples per chunk.
    ui64 nBlocks = nSim / CHUNK;
    ui64 reste = nSim % CHUNK;

    double payoffSum = 0.0;

    #pragma omp parallel reduction(+:payoffSum)
    {
        xorshift128plus_state rng;
        ui64 seed = 0xFACEBEEF ^ (0x1234ULL * (omp_get_thread_num() + 1));
        xorshift128plus_init(rng, seed);

        std::vector<double> gArr(CHUNK);

        for (ui64 b = 0; b < nBlocks; b++) {
            // Generate CHUNK Gaussian samples.
            for (int i = 0; i < CHUNK; i += 2) {
                ui64 rA = xorshift128plus(rng);
                ui64 rB = xorshift128plus(rng);
                double u1 = double(rA) * (1.0 / 18446744073709551616.0);
                double u2 = double(rB) * (1.0 / 18446744073709551616.0);
                double ga, gb;
                box_muller_2(u1, u2, ga, gb);

                gArr[i] = ga;
                if (i + 1 < CHUNK) {
                    gArr[i + 1] = gb;
                }
            }

            // Vectorized payoff computation.
            svbool_t pgall = svptrue_b64();
            int VL = svcntd(); // Number of doubles per vector.
            double sumLocal = 0.0;
            for (int i = 0; i < CHUNK; i += VL) {
                svbool_t pg = svwhilelt_b64(i, CHUNK); // Predicate for active lanes.
                svfloat64_t vg = svld1_f64(pg, &gArr[i]);

                svfloat64_t x = svmad_f64_m(pg, vg, svdup_f64(vol), svdup_f64(drift));
                svfloat64_t ex = sve_exp_approx(x);
                svfloat64_t vst = svmul_f64_m(pg, ex, svdup_f64(S0));
                svfloat64_t pay = sve_payoff(vst, svdup_f64(K), pg);

                double tmp[64];
                svst1_f64(pg, tmp, pay);
                for (int j = 0; j < VL && (i + j < CHUNK); j++) {
                    sumLocal += tmp[j];
                }
            }
            payoffSum += sumLocal;
        }

        // Handle remainder simulations.
        if(reste>0){
            std::vector<double> gR(reste);
            for(ui64 i=0; i<reste; i+=2){
                ui64 ra=xorshift128plus(rng);
                ui64 rb=xorshift128plus(rng);
                double u1= double(ra)*(1.0/18446744073709551616.0);
                double u2= double(rb)*(1.0/18446744073709551616.0);
                double ga,gb;
                box_muller_2(u1,u2,ga,gb);
                gR[i]= ga;
                if(i+1<reste) gR[i+1]= gb;
            }
            double sumLocal=0.0;
            // On re-vec
            int VL= svcntd();
            for(ui64 i=0; i<reste; i+=VL){
                svbool_t pg= svwhilelt_b64(i, reste);
                if(!svptest_any(svptrue_b64(), pg)) break;
                svfloat64_t vg= svld1_f64(pg, &gR[i]);
                svfloat64_t x= svmad_f64_m(pg, vg, svdup_f64(vol),
                                                svdup_f64(drift));
                svfloat64_t ex= sve_exp_approx(x);
                svfloat64_t vst= svmul_f64_m(pg, ex, svdup_f64(S0));
                svfloat64_t pay= sve_payoff(vst, svdup_f64(K), pg);

                double tmp[64];
                svst1_f64(pg, tmp, pay);
                for(int j=0; j<VL && (i+j<reste); j++){
                    sumLocal+= tmp[j];
                }
            }
    return payoffSum / double(nSim) * disc;
}

/*******************************************
 * @brief Main function for Monte Carlo SVE Black-Scholes simulation.
 *
 * @param argc Argument count.
 * @param argv Argument values (num_sims and num_runs).
 * @return Execution status.
 *******************************************/
int main(int argc, char* argv[])
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <num_sims> <num_runs>\n";
        return 1;
    }
    ui64 nSim = std::stoull(argv[1]);
    ui64 nRuns = std::stoull(argv[2]);

    double S0 = 100.0, K = 110.0, T = 1.0, r = 0.06, sigma = 0.2;

    double t1 = dml_micros();
    double tot = 0.0;
    for (ui64 run = 0; run < nRuns; run++) {
        tot += black_scholes_monte_carlo_svechunk(S0, K, T, r, sigma, nSim);
    }
    double t2 = dml_micros();

    double meanVal = tot / double(nRuns);
    double elap = (t2 - t1) * 1e-6;
    std::cout << std::fixed << std::setprecision(6)
              << "SVE-chunk => value= " << meanVal
              << " in " << elap << " s\n";
    return 0;
}
