#ifndef CFR_PLUS_H_
#define CFR_PLUS_H_

#include "./lookahead.h"

void calculate_strategy(LookaheadNode* node, const double cum_strat_weight);
void calculate_counterfactual_values(LookaheadNode* node, const AssignmentProbs& starting_probs);

void calculate_cumulative_strategy(LookaheadNode* root);

void cfr_get_values(
    LookaheadNode* root,
    const int iterations,
    const int wait_iterations,
    const AssignmentProbs& starting_probs,
    const bool save_strategy,
    ViewpointVector* values
);

#endif // CFR_PLUS_H_
