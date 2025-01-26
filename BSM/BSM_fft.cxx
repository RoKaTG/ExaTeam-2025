#include <iostream>
#include <amath.h>
#include <armpl.h>
#include <vector>
#include <complex>
#include <cstdlib>
#include <omp.h>

/*******************************************
 * @brief Characteristic function for the 
 * Black-Scholes model using OpenMP.
 * 
 * @param u Fourier domain variable
 * @param T Time to maturity
 * @param r Risk-free interest rate
 * @param sigma Volatility
 * @param S0 Initial stock price
 * @return Complex value of the characteristic function
 *******************************************/
std::complex<double> charfunc_black_scholes_omp(
    double u, double T, double r, double sigma, double S0)
{
    std::complex<double> i(0.0, 1.0); // Imaginary unit
    double mu = std::log(S0) + (r - 0.5 * sigma * sigma) * T; // Drift term
    std::complex<double> ex = i * (u * mu) - 0.5 * sigma * sigma * u * u * T; // Exponent
    return std::exp(ex); // Return characteristic function
}

/*******************************************
 * @brief Compute option price using the Carr-Madan method 
 * with FFT and OpenMP parallelization.
 *
 * @param S0 Initial stock price
 * @param K Strike price
 * @param T Time to maturity
 * @param r Risk-free interest rate
 * @param sigma Volatility
 * @param alpha Damping factor
 * @param eta Step size in Fourier space
 * @param N Number of FFT points
 * @return Call option price
 *******************************************/
double carr_madan_fft_price_omp(
    double S0, double K, double T, double r, double sigma,
    double alpha, double eta, int N)
{
    std::vector<std::complex<double>> inFFT(N), outFFT(N); // Input and output arrays for FFT
    double lnK = std::log(K); // Logarithm of strike price

    // Parallelized pre-computation of the input array using OpenMP
    #pragma omp parallel for
    for (int k = 0; k < N; k++) {
        double u = k * eta; // Fourier domain variable
        std::complex<double> iC(0.0, 1.0); // Imaginary unit
        std::complex<double> phi = charfunc_black_scholes_omp(u - iC * alpha, T, r, sigma, S0); // Characteristic function

        std::complex<double> denom = (alpha + iC * u); // Denominator
        std::complex<double> num = phi * std::exp(-r * T) * std::exp(iC * u * lnK); // Numerator

        if (std::abs(denom.real()) < 1e-14 && std::abs(denom.imag()) < 1e-14) {
            inFFT[k] = 0.0; // Handle numerical instability
        } else {
            inFFT[k] = num / denom; // Compute input value
        }
    }

    // Prepare FFTW
    fftw_complex* fin = reinterpret_cast<fftw_complex*>(inFFT.data());
    fftw_complex* fout = reinterpret_cast<fftw_complex*>(outFFT.data());

    fftw_plan plan = fftw_plan_dft_1d(N, fin, fout, FFTW_BACKWARD, FFTW_ESTIMATE); // FFTW plan

    // Optional: Enable FFTW multi-threading if compiled with thread support
    // fftw_init_threads();
    // fftw_plan_with_nthreads(omp_get_max_threads());

    fftw_execute(plan); // Execute FFT
    fftw_destroy_plan(plan); // Destroy FFTW plan

    // Normalize output
    std::complex<double> val0 = outFFT[0] / (double)N; // FFT normalization
    double c = std::exp(-alpha * lnK) / M_PI; // Scaling factor
    double price = c * val0.real() * eta; // Final option price

    return price;
}

/*******************************************
 * @brief Main function to run the Carr-Madan FFT pricing.
 *
 * @param argc Argument count
 * @param argv Argument values (optional parameters for S0, K, T, r, sigma, alpha, eta, N)
 * @return Execution status
 *******************************************/
int main(int argc, char* argv[])
{
    // Default parameters
    double S0 = 100.0, K = 100.0, T = 1.0, r = 0.06, sigma = 0.2;
    double alpha = 1.5, eta = 0.1;
    int N = 4096;

    // Parse optional command-line arguments
    if (argc > 1) S0 = std::atof(argv[1]);
    if (argc > 2) K = std::atof(argv[2]);
    if (argc > 3) T = std::atof(argv[3]);
    if (argc > 4) r = std::atof(argv[4]);
    if (argc > 5) sigma = std::atof(argv[5]);
    if (argc > 6) alpha = std::atof(argv[6]);
    if (argc > 7) eta = std::atof(argv[7]);
    if (argc > 8) N = std::atoi(argv[8]);

    // Measure computation time
    double t0 = omp_get_wtime();
    double price = carr_madan_fft_price_omp(S0, K, T, r, sigma, alpha, eta, N);
    double t1 = omp_get_wtime();

    // Output results
    std::cout << "Carr-Madan FFT Price (OMP), call= " << price
              << "  (threads=" << omp_get_max_threads() << ")\n"
              << "Elapsed= " << (t1 - t0) << "s\n";

    return 0;
}

