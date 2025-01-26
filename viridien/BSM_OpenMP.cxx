#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <limits>
#include <algorithm>
#include <iomanip>   // For setting precision
#include <omp.h>     // OpenMP header

#define ui64 u_int64_t

#include <sys/time.h>
double dml_micros()
{
        static struct timezone tz;
        static struct timeval  tv;
        gettimeofday(&tv,&tz);
        return((tv.tv_sec*1000000.0)+tv.tv_usec);
}

// Function to generate Gaussian noise using Box-Muller transform
double gaussian_box_muller() {
    static thread_local std::mt19937 generator(std::random_device{}());
    static thread_local std::normal_distribution<double> distribution(0.0, 1.0);
    return distribution(generator);
}

std::vector<double> gaussian_box_muller_batch(size_t n) {
    static thread_local std::mt19937 generator(std::random_device{}());
    static thread_local std::normal_distribution<double> distribution(0.0, 1.0);

    std::vector<double> values(n);
    for (size_t i = 0; i < n; ++i) {
        values[i] = distribution(generator);
    }
    return values;
}

double black_scholes_monte_carlo(uint64_t S0, uint64_t K, double T, double r, double sigma, double q, uint64_t num_simulations) {
    // Pre-compute constants
    const double drift = (r - q - 0.5 * sigma * sigma) * T;
    const double diffusion = sigma * sqrt(T);
    const double discount_factor = exp(-r * T);

    double sum_payoffs = 0.0;

    // Batch size for vectorization
    const size_t batch_size = 2;  // Adjust this value based on SIMD width
    const size_t num_batches = num_simulations / batch_size;
    const size_t remainder = num_simulations % batch_size;

    // Monte Carlo simulation loop (vectorized by batches)
    #pragma omp parallel reduction(+:sum_payoffs)
    {
        std::vector<double> Z(batch_size);
        #pragma omp for
        for (size_t batch = 0; batch < num_batches; ++batch) {
            Z = gaussian_box_muller_batch(batch_size);
            double batch_sum = 0.0;

            #pragma omp simd reduction(+:batch_sum)
            for (size_t i = 0; i < batch_size; ++i) {
                const double ST = S0 * exp(drift + diffusion * Z[i]);  // Simulate stock price at maturity
                const double payoff = std::max(ST - K, 0.0);          // Call option payoff
                batch_sum += payoff;
            }
            sum_payoffs += batch_sum;
        }

        // Handle remainder (non-vectorized)
        #pragma omp single
        {
            if (remainder > 0) {
                std::vector<double> Z_rem = gaussian_box_muller_batch(remainder);
                for (size_t i = 0; i < remainder; ++i) {
                    const double ST = S0 * exp(drift + diffusion * Z_rem[i]);
                    const double payoff = std::max(ST - K, 0.0);
                    sum_payoffs += payoff;
                }
            }
        }
    }

    return discount_factor * (sum_payoffs / num_simulations);
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <num_simulations> <num_runs>" << std::endl;
        return 1;
    }

    ui64 num_simulations = std::stoull(argv[1]);
    ui64 num_runs        = std::stoull(argv[2]);

    // Input parameters
    ui64 S0      = 100;                   // Initial stock price
    ui64 K       = 110;                   // Strike price
    double T     = 1.0;                   // Time to maturity (1 year)
    double r     = 0.06;                  // Risk-free interest rate
    double sigma = 0.2;                   // Volatility
    double q     = 0.03;                  // Dividend yield

    // Generate a random seed at the start of the program using random_device
    std::random_device rd;
    unsigned long long global_seed = rd();  // This will be the global seed
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    std::cout << "Global initial seed: " << global_seed << "      argv[1]= " << argv[1] << "     argv[2]= " << argv[2] << "   Num threads: " << num_threads << std::endl;

    double sum=0.0;
    double t1=dml_micros();
    #pragma omp parallel for reduction(+:sum)
    for (ui64 run = 0; run < num_runs; ++run) {
        sum+= black_scholes_monte_carlo(S0, K, T, r, sigma, q, num_simulations);
    }
    double t2=dml_micros();
    std::cout << std::fixed << std::setprecision(6) << " value= " << sum/num_runs << " in " << (t2-t1)/1000000.0 << " seconds" << std::endl;

    return 0;
}
