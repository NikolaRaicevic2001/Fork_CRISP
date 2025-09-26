#pragma once

#include "solver_core/SolverInterface.h"

#include <filesystem>
#include <chrono>
#include <random>   
#include "math.h"

using namespace CRISP;

// Helper Functions
static inline double clamp(double v, double lo, double hi){ return std::max(lo, std::min(hi, v));}

// Function that generates a random vector of size num_state+num_control and then repeats it N times to form the initial guess
vector_t makeRandomFirstGuess(const size_t N, const size_t num_state, const size_t num_control, const scalar_t a, const scalar_t b, const unsigned seed = 40)
{ 
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U01(0.0, 1.0);
    std::normal_distribution<double> N01(0.0, 1.0);

    // Initial Guess Vector
    vector_t x(N*(num_state + num_control));

    x.setZero();

    // Random vector of size num_state+num_control
    vector_t random_vector(num_state + num_control);
    random_vector.setZero();

    random_vector[0] = U01(rng) * 4.0 - 2.0;   // px in [-2, 2]
    random_vector[1] = U01(rng) * 4.0 - 2.0;   // py in [-2, 2]
    random_vector[2] = U01(rng) * 1.0 * M_PI;  // theta in [0, pi]

    // Random contact point inside box with a small safety margin
    double edge_margin = 1e-3;
    double cx = (2.0 * U01(rng) - 1.0) * (a - edge_margin);
    double cy = (2.0 * U01(rng) - 1.0) * (b - edge_margin);
    random_vector[3] = clamp(cx, -a + edge_margin,  a - edge_margin);   // cx
    random_vector[4] = clamp(cy, -b + edge_margin,  b - edge_margin);   // cy

    // Random force values
    random_vector[5] = N01(rng);   // lambda1
    random_vector[6] = N01(rng);   // lambda2
    random_vector[7] = N01(rng);   // lambda3
    random_vector[8] = N01(rng);   // lambda4

    // Print the generated random vector
    std::cout << "Generated random vector: " << random_vector.transpose() << std::endl;
    
    // Repeat the random vector N times to form the initial guess
    for (size_t i = 0; i < N; ++i) {
        x.segment(i * (num_state + num_control), num_state + num_control) = random_vector; 
    }

    return x;
}
