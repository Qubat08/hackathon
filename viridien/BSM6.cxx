#include <iostream>
#include <cmath>
#include <vector>
#include <iomanip>
#include <omp.h>
#include <sys/time.h>
#include <random>
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
    const Real p1 = (r - q - 0.5f * sigma * sigma) * T;
    const Real p2 = sigma * real_sqrt(T);
    const Real p3 = exp(-r * T);
    
    Real sum_payoffs = 0.0f;

    #pragma omp parallel reduction(+:sum_payoffs)
    {
        uint64_t thread_seed = seed + omp_get_thread_num();
        XoshiroCpp::Xoshiro256Plus generator(thread_seed);
        std::normal_distribution<Real> distribution(0.0, 1.0);

        Real Z[4];
        #pragma omp for schedule(static)
        for (size_t i = 0; i < num_simulations; i += 4) {
            for (size_t j = 0; j < 4; ++j) {
                Z[j] = distribution(generator);
            }

            for (size_t j = 0; j < 4; ++j) {
                Real ST = S0 * exp(p1 + p2 * Z[j]);
                Real payoff = std::max(ST - K, 0.0f);
                sum_payoffs += payoff;
            }
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

    std::random_device rd;
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
