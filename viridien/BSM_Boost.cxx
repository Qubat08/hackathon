#include <iostream>
#include <vector>
#include <omp.h>
#include <sys/time.h>
#include <boost/random.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/nondet_random.hpp>
#include <experimental/simd>
#include <armpl.h>
#include "XoshiroCpp.hpp"

using ui64 = uint64_t; // Typedef pour simplifier
using Real = float;    // Utilisation de using au lieu de define

inline Real real_sqrt(Real x) {
    if constexpr (std::is_same_v<Real, double>) {
        return sqrt(x);
    } else if constexpr (std::is_same_v<Real, float>) {
        return sqrtf(x);
    } else {
        return sqrtl(x);
    }
}

double dml_micros() {
    static struct timezone tz;
    static struct timeval tv;
    gettimeofday(&tv, &tz);
    return ((tv.tv_sec * 1000000.0) + tv.tv_usec);
}

Real black_scholes_monte_carlo(uint64_t S0, uint64_t K, Real T, Real r, Real sigma, Real q, uint64_t num_simulations, uint64_t seed) {
    using simd_t = std::experimental::native_simd<Real>;
    simd_t p1 = (r - q - 0.5f * sigma * sigma) * T;
    simd_t p2 = sigma * std::sqrt(T);
    Real p3 = exp(-r * T);

    Real sum_payoffs = 0.0f;

    #pragma omp parallel reduction(+:sum_payoffs) firstprivate(p1, p2, p3)
    {
        uint64_t thread_seed = seed + omp_get_thread_num();
        XoshiroCpp::Xoshiro256Plus generator(thread_seed);
        boost::random::normal_distribution<Real> distribution(0.0, 1.0);

        #pragma omp for
        for (size_t i = 0; i < num_simulations; i += simd_t::size()) {
            simd_t Z;
            for (size_t j = 0; j < simd_t::size(); ++j) {
                Z[j] = distribution(generator);
            }

            simd_t S0_simd = simd_t(static_cast<Real>(S0));
            simd_t K_simd = simd_t(static_cast<Real>(K));
            simd_t ST = S0_simd * std::experimental::exp(p1 + p2 * Z);
            simd_t payoff = std::experimental::max(ST - K_simd, simd_t(0.0f));

            sum_payoffs += std::experimental::reduce(payoff);
        }
    }

    return p3 * (sum_payoffs / num_simulations);
}




int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_simulations> <num_runs>" << std::endl;
        return 1;
    }

    uint64_t num_simulations = std::stoull(argv[1]);
    uint64_t num_runs = std::stoull(argv[2]);

    uint64_t S0 = 100;
    uint64_t K = 110;
    Real T = 1.0f;
    Real r = 0.06f;
    Real sigma = 0.2f;
    Real q = 0.03f;

    boost::random_device rd;
    uint64_t global_seed = rd();
    int num_threads;

    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    std::cout << "Global initial seed: " << global_seed
              << "      argv[1]= " << argv[1]
              << "     argv[2]= " << argv[2]
              << "   Num threads: " << num_threads << std::endl;

    Real sum = 0.0f;
    double t1 = dml_micros();

    #pragma omp parallel for reduction(+:sum)
    for (uint64_t run = 0; run < num_runs; ++run) {
        sum += black_scholes_monte_carlo(S0, K, T, r, sigma, q, num_simulations, global_seed + run);
    }

    double t2 = dml_micros();
    std::cout << std::fixed << std::setprecision(6) << " value= " << sum / num_runs
              << " in " << (t2 - t1) / 1000000.0 << " seconds" << std::endl;

    return 0;
}
