#ifndef UTIL_H_
#define UTIL_H_

#include <string>
#include <random>

#include "./lookahead.h"

extern std::mt19937 rng;

struct Initialization {
    int depth;
    int num_succeeds;
    int num_fails;
    int propose_count;

    int proposer;
    std::string generate_start_technique;
    AssignmentProbs starting_probs;

    int iterations;
    int wait_iterations;
    ViewpointVector solution_values[NUM_PLAYERS];

    std::string Stringify() const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

void seed_rng();

void prepare_initialization(
    const int depth,
    const int num_succeeds,
    const int num_fails,
    const int propose_count,
    Initialization* init
);

void run_initialization_with_cfr(
    const int iterations,
    const int wait_iterations,
    const std::string& model_search_dir,
    Initialization* init
);

#endif // UTIL_H_
