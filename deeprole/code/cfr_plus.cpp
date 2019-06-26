#include <iostream>
#include <memory>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <array>
#include <utility>

#include "cfr_plus.h"
#include "lookup_tables.h"

using namespace std;

#define ASSERT(left,operator,right) { if(!((left) operator (right))){ std::cerr << "ASSERT FAILED: " << #left << #operator << #right << " @ " << __FILE__ << " (" << __LINE__ << "). " << #left << "=" << (left) << "; " << #right << "=" << (right) << std::endl; } }


static double my_single_pass_responsibility(LookaheadNode* node, int me, int my_viewpoint, int my_partner) {
    assert(node->type == MISSION);
    assert(VIEWPOINT_TO_BAD[me][my_viewpoint] == my_partner);
    assert(my_viewpoint >= NUM_GOOD_VIEWPOINTS);
    int partner_viewpoint = VIEWPOINT_TO_PARTNER_VIEWPOINT[me][my_viewpoint];
    assert(VIEWPOINT_TO_PARTNER_VIEWPOINT[my_partner][partner_viewpoint] == my_viewpoint);
    assert(node->proposal & (1 << me));
    assert(node->proposal & (1 << my_partner));
    double my_pass_prob = node->mission_strategy->at(me)(my_viewpoint, 0);
    double partner_pass_prob = node->mission_strategy->at(my_partner)(partner_viewpoint, 0);
    double outcome_prob = my_pass_prob * (1.0 - partner_pass_prob) + (1.0 - my_pass_prob) * partner_pass_prob;
    double my_responsibility_portion = my_pass_prob * my_pass_prob + (1.0 - my_pass_prob) * (1.0 - my_pass_prob);
    double partner_responsibility_portion = partner_pass_prob * partner_pass_prob + (1.0 - partner_pass_prob) * (1.0 - partner_pass_prob);
    double my_responsibility_exponent = my_responsibility_portion / (my_responsibility_portion + partner_responsibility_portion);
    return pow(outcome_prob, my_responsibility_exponent);
}

static void add_middle_cfvs(LookaheadNode* node, int me, int my_viewpoint, int my_partner, double* pass_cfv, double* fail_cfv) {
    assert(node->type == MISSION);
    assert(VIEWPOINT_TO_BAD[me][my_viewpoint] == my_partner);
    assert(my_viewpoint >= NUM_GOOD_VIEWPOINTS);
    int partner_viewpoint = VIEWPOINT_TO_PARTNER_VIEWPOINT[me][my_viewpoint];
    assert(VIEWPOINT_TO_PARTNER_VIEWPOINT[my_partner][partner_viewpoint] == my_viewpoint);
    assert(node->proposal & (1 << me));
    assert(node->proposal & (1 << my_partner));

    double partner_responsibility = my_single_pass_responsibility(node, my_partner, partner_viewpoint, me);
    double middle_cfv = node->children[1]->counterfactual_values[me](my_viewpoint);
    middle_cfv /= partner_responsibility;

    #ifndef NDEBUG
    double my_prob = node->mission_strategy->at(me)(my_viewpoint, 0);
    double partner_prob = node->mission_strategy->at(my_partner)(partner_viewpoint, 0);
    double my_resp = my_single_pass_responsibility(node, me, my_viewpoint, my_partner);
    double partner_resp = my_single_pass_responsibility(node, my_partner, partner_viewpoint, me);
    double combined_resp = my_resp * partner_resp;
    double expected_resp = (1.0 - my_prob) * partner_prob + my_prob * (1.0 - partner_prob);

    if (abs(combined_resp - expected_resp) > 1e-10) {
        std::cerr << "combined_resp: " << combined_resp << endl;
        std::cerr << "expected_resp: " << expected_resp << endl;
        std::cerr << "   difference: " << abs(combined_resp - expected_resp) << endl;
        assert(false);
    }

    double middle_cfv_2 = node->children[1]->counterfactual_values[me](my_viewpoint) * my_resp;
    middle_cfv_2 /= expected_resp;
    if (abs(middle_cfv - middle_cfv_2) > 1e-10) {
        std::cerr << "middle_cfv: " << middle_cfv << endl;
        std::cerr << "middle_cfv_2: " << middle_cfv_2 << endl;
        std::cerr << "   difference: " << abs(middle_cfv - middle_cfv_2) << endl;
        assert(false);
    }
    #endif

    double partner_pass_prob = node->mission_strategy->at(my_partner)(partner_viewpoint, 0);
    *pass_cfv += middle_cfv * (1.0 - partner_pass_prob);
    *fail_cfv += middle_cfv * partner_pass_prob;
}

static void fill_reach_probabilities_for_mission_node(LookaheadNode* node) {
    assert(node->type == MISSION);

    for (int fails = 0; fails < NUM_EVIL + 1; fails++) {
        auto& child = node->children[fails];
        // First, pass-through all of the players not on the mission.
        // Second, pass-through all of the viewpoints where the player is good.
        //      - this step won't work on it's own, we need to fix things at the leaf node to ensure we remove possibilities where a good player was forced to fail.
        // This is optimized - we'll be multiplying "correctly" later down in this function.
        for (int player = 0; player < NUM_PLAYERS; player++) {
            child->reach_probs[player] = node->reach_probs[player];
        }

        switch (fails) {
        case 0: {
            // No one failed.
            // For evil players, their move probability gets multiplied by the pass probability.
            for (int player = 0; player < NUM_PLAYERS; player++) {
                // Skip players not on the mission.
                if (((1 << player) & node->proposal) == 0) continue;
                for (int viewpoint = NUM_GOOD_VIEWPOINTS; viewpoint < NUM_VIEWPOINTS; viewpoint++) {
                    child->reach_probs[player](viewpoint) *= node->mission_strategy->at(player)(viewpoint, 0);
                }
            }
        } break;
        case 1: {
            // This is the hard case, since we're combining probabilities oddly.
            for (int player = 0; player < NUM_PLAYERS; player++) {
                // Skip players not on the mission.
                if (((1 << player) & node->proposal) == 0) continue;
                for (int viewpoint = NUM_GOOD_VIEWPOINTS; viewpoint < NUM_VIEWPOINTS; viewpoint++) {
                    int player_partner = VIEWPOINT_TO_BAD[player][viewpoint];
                    if ((1 << player_partner) & node->proposal) {
                        // The player's partner is on the mission. Weird stuff!
                        child->reach_probs[player](viewpoint) *= my_single_pass_responsibility(node, player, viewpoint, player_partner);
                    } else {
                        // The player's partner is not on the mission, fail normally.
                        child->reach_probs[player](viewpoint) *= node->mission_strategy->at(player)(viewpoint, 1);
                    }
                }
            }
        } break;
        case 2: {
            // Everyone failed.
            // For evil players, their move probability gets multiplied by the fail probability.
            for (int player = 0; player < NUM_PLAYERS; player++) {
                // Skip players not on the mission.
                if (((1 << player) & node->proposal) == 0) continue;
                for (int viewpoint = NUM_GOOD_VIEWPOINTS; viewpoint < NUM_VIEWPOINTS; viewpoint++) {
                    child->reach_probs[player](viewpoint) *= node->mission_strategy->at(player)(viewpoint, 1);
                }
            }
        } break;
        }
    }
}

static void fill_reach_probabilities(LookaheadNode* node) {
    switch (node->type) {
    case PROPOSE: {
        int player = node->proposer;
        for (int proposal = 0; proposal < NUM_PROPOSAL_OPTIONS; proposal++) {
            auto& child = node->children[proposal];
            for (int i = 0; i < NUM_PLAYERS; i++) {
                child->reach_probs[i] = node->reach_probs[i];
            }

            child->reach_probs[player] *= node->propose_strategy->col(proposal);
        }
    } break;
    case VOTE: {
        for (uint32_t vote_pattern = 0; vote_pattern < (1 << NUM_PLAYERS); vote_pattern++) {
            auto& child = node->children[vote_pattern];
            for (int player = 0; player < NUM_PLAYERS; player++) {
                int vote = (vote_pattern >> player) & 1;
                child->reach_probs[player] = node->reach_probs[player] * node->vote_strategy->at(player).col(vote);
            }
        }
    } break;
    case MISSION: {
        fill_reach_probabilities_for_mission_node(node);
    } break;
    default: break;
    }
}

static void populate_full_reach_probs(LookaheadNode* node) {
    for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
        double probability = 1.0;
        int evil = ASSIGNMENT_TO_EVIL[i];
        for (auto fail : node->fails) {
            if (__builtin_popcount(fail.first & evil) < fail.second) {
                probability = 0.0;
                break;
            }
        }
        for (int player = 0; player < NUM_PLAYERS && probability != 0; player++) {
            int viewpoint = ASSIGNMENT_TO_VIEWPOINT[i][player];
            probability *= node->reach_probs[player](viewpoint);
        }
        (*(node->full_reach_probs))(i) = probability;
    }
}

void calculate_strategy(LookaheadNode* node, const double cum_strat_weight) {
    switch (node->type) {
    case PROPOSE: {
        // Initialize the node's memory
        auto& player_regrets = *(node->propose_regrets);
        auto& player_strategy = *(node->propose_strategy);
        #ifdef CFR_PLUS
        // These are already maxed
        player_strategy = player_regrets;
        #else
        // These aren't so we have to max them.
        player_strategy = player_regrets.max(0.0);
        #endif

        ProposeData tmp_holder = player_strategy.colwise() / player_strategy.rowwise().sum();
        tmp_holder = tmp_holder.unaryExpr([](double v) { return std::isfinite(v) ? v : 1.0/NUM_PROPOSAL_OPTIONS; });
        player_strategy = (1.0 - TREMBLE_VALUE) * tmp_holder + TREMBLE_VALUE * ProposeData::Constant(1.0/NUM_PROPOSAL_OPTIONS);

        if (cum_strat_weight != 0) {
            *(node->propose_cum) += (player_strategy.colwise() * node->reach_probs[node->proposer]) * cum_strat_weight;
        }
    } break;
    case VOTE: {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            auto& player_regrets = node->vote_regrets->at(i);
            auto& player_strategy = node->vote_strategy->at(i);

            #ifdef CFR_PLUS
            // These are already maxed
            player_strategy = player_regrets;
            #else
            // These aren't so we have to max them.
            player_strategy = player_regrets.max(0.0);
            #endif

            VoteData tmp_holder = player_strategy.colwise() / player_strategy.rowwise().sum();
            tmp_holder = tmp_holder.unaryExpr([](double v) { return std::isfinite(v) ? v : 0.5; });
            player_strategy = (1.0 - TREMBLE_VALUE) * tmp_holder + TREMBLE_VALUE * VoteData::Constant(0.5);

            if (cum_strat_weight != 0) {
                node->vote_cum->at(i) += (player_strategy.colwise() * node->reach_probs[i]) * cum_strat_weight;
            }
        }
    } break;
    case MISSION: {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            if (((1 << i) & node->proposal) == 0) continue;
            auto& player_regrets = node->mission_regrets->at(i);
            auto& player_strategy = node->mission_strategy->at(i);

            #ifdef CFR_PLUS
            // These are already maxed
            player_strategy = player_regrets;
            #else
            // These aren't so we have to max them.
            player_strategy = player_regrets.max(0.0);
            #endif

            MissionData tmp_holder = player_strategy.colwise() / player_strategy.rowwise().sum();
            tmp_holder = tmp_holder.unaryExpr([](double v) { return std::isfinite(v) ? v : 0.5; });
            player_strategy = (1.0 - TREMBLE_VALUE) * tmp_holder + TREMBLE_VALUE * MissionData::Constant(0.5);

            if (cum_strat_weight != 0) {
                node->mission_cum->at(i) += (player_strategy.colwise() * node->reach_probs[i]) * cum_strat_weight;
            }
        }
    } break;
    case TERMINAL_MERLIN: {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            auto& player_regrets = node->merlin_regrets->at(i);
            auto& player_strategy = node->merlin_strategy->at(i);

            #ifdef CFR_PLUS
            // These are already maxed
            player_strategy = player_regrets;
            #else
            // These aren't so we have to max them.
            player_strategy = player_regrets.max(0.0);
            #endif

            MerlinData tmp_holder = player_strategy.colwise() / player_strategy.rowwise().sum();
            tmp_holder = tmp_holder.unaryExpr([](double v) { return std::isfinite(v) ? v : 1.0/NUM_PLAYERS; });
            player_strategy = (1.0 - TREMBLE_VALUE) * tmp_holder + TREMBLE_VALUE * MerlinData::Constant(1.0/NUM_PLAYERS);

            if (cum_strat_weight != 0) {
                node->merlin_cum->at(i) += (player_strategy.colwise() * node->reach_probs[i]) * cum_strat_weight;
            }
        }
        // Intentional missing break.
    }
    case TERMINAL_NO_CONSENSUS:
    case TERMINAL_TOO_MANY_FAILS:
    case TERMINAL_PROPOSE_NN: {
        populate_full_reach_probs(node);
    } break;
    default: break;
    }

    fill_reach_probabilities(node);

    for (auto& child : node->children) {
        calculate_strategy(child.get(), cum_strat_weight);
    }
}

static void calculate_propose_cfvs(LookaheadNode* node) {
    for (int player = 0; player < NUM_PLAYERS; player++) {
        for (int proposal = 0; proposal < NUM_PROPOSAL_OPTIONS; proposal++) {
            auto& child = node->children[proposal];
            if (player == node->proposer) {
                node->counterfactual_values[player] += child->counterfactual_values[player] * node->propose_strategy->col(proposal);
            } else {
                node->counterfactual_values[player] += child->counterfactual_values[player];
            }
        }
    }

    // Update regrets
    for (int proposal = 0; proposal < NUM_PROPOSAL_OPTIONS; proposal++) {
        node->propose_regrets->col(proposal) += node->children[proposal]->counterfactual_values[node->proposer] - node->counterfactual_values[node->proposer];
    }
    #ifdef CFR_PLUS
    *(node->propose_regrets) = node->propose_regrets->max(0.0);
    #endif
}

static void calculate_vote_cfvs(LookaheadNode* node) {
    for (int player = 0; player < NUM_PLAYERS; player++) {
        VoteData cfvs = VoteData::Constant(0.0);

        for (int vote_pattern = 0; vote_pattern < (1 << NUM_PLAYERS); vote_pattern++) {
            auto& child = node->children[vote_pattern];
            int vote = (vote_pattern >> player) & 1;
            cfvs.col(vote) += child->counterfactual_values[player];
        }

        // Update regrets
        node->counterfactual_values[player] = (cfvs * node->vote_strategy->at(player)).rowwise().sum();
        node->vote_regrets->at(player) += cfvs.colwise() - node->counterfactual_values[player];
        #ifdef CFR_PLUS
        node->vote_regrets->at(player) = node->vote_regrets->at(player).max(0.0);
        #endif
    }
}

static void calculate_mission_cfvs(LookaheadNode* node) {
    // For players not on the mission, the CFVs are just the sum.
    for (int player = 0; player < NUM_PLAYERS; player++) {
        // Skip players on the mission
        if ((1 << player) & node->proposal) continue;

        for (int num_fails = 0; num_fails < NUM_EVIL + 1; num_fails++) {
            node->counterfactual_values[player] += node->children[num_fails]->counterfactual_values[player];
        }
    }

    // For players on the mission, the CFVs are a little more complicated.
    for (int player = 0; player < NUM_PLAYERS; player++) {
        // Skip players not on the mission.
        if (((1 << player) & node->proposal) == 0) continue;

        // For good viewpoints, CFVs are just the sum of the number of possible fails
        for (int viewpoint = 0; viewpoint < NUM_GOOD_VIEWPOINTS; viewpoint++) {
            for (int num_fails = 0; num_fails < NUM_EVIL + 1; num_fails++) {
                node->counterfactual_values[player](viewpoint) += node->children[num_fails]->counterfactual_values[player](viewpoint);
            }   
        }

        // For bad viewpoints, CFVs are split.
        for (int viewpoint = NUM_GOOD_VIEWPOINTS; viewpoint < NUM_VIEWPOINTS; viewpoint++) {
            double pass_cfv = 0.0;
            double fail_cfv = 0.0;
            int partner = VIEWPOINT_TO_BAD[player][viewpoint];
            if ((1 << partner) & node->proposal) {
                // The partner is on the mission.
                pass_cfv = node->children[0]->counterfactual_values[player](viewpoint);
                fail_cfv = node->children[2]->counterfactual_values[player](viewpoint);
                add_middle_cfvs(node, player, viewpoint, partner, &pass_cfv, &fail_cfv);
            } else {
                // The partner is not on the mission. CFVs are "simple" - 0 or 1 fails possible.
                assert(node->children[2]->counterfactual_values[player](viewpoint) == 0.0);
                pass_cfv = node->children[0]->counterfactual_values[player](viewpoint);
                fail_cfv = node->children[1]->counterfactual_values[player](viewpoint);
            }
            double my_pass_prob = node->mission_strategy->at(player)(viewpoint, 0);
            double result_cfv = pass_cfv * my_pass_prob + fail_cfv * (1.0 - my_pass_prob);
            node->counterfactual_values[player](viewpoint) = result_cfv;
            node->mission_regrets->at(player)(viewpoint, 0) += pass_cfv - result_cfv;
            node->mission_regrets->at(player)(viewpoint, 1) += fail_cfv - result_cfv;
        }
        #ifdef CFR_PLUS
        node->mission_regrets->at(player) = node->mission_regrets->at(player).max(0.0);
        #endif
    }
}

static void calculate_merlin_cfvs(LookaheadNode* node, const AssignmentProbs& starting_probs) {
    for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
        int merlin = ASSIGNMENT_TO_ROLES[i][0];
        int assassin = ASSIGNMENT_TO_ROLES[i][1];
        int assassin_viewpoint = ASSIGNMENT_TO_VIEWPOINT[i][assassin];
        int evil = ASSIGNMENT_TO_EVIL[i];
        double reach_prob = (*(node->full_reach_probs))(i) * starting_probs(i);
        if (reach_prob == 0.0) continue;

        double correct_prob = node->merlin_strategy->at(assassin)(assassin_viewpoint, merlin);

        for (int player = 0; player < NUM_PLAYERS; player++) {
            int viewpoint = ASSIGNMENT_TO_VIEWPOINT[i][player];
            double counterfactual_reach_prob = reach_prob / node->reach_probs[player](viewpoint);
            if ((1 << player) & evil) {
                node->counterfactual_values[player](viewpoint) += counterfactual_reach_prob * (
                    EVIL_WIN_PAYOFF * correct_prob +
                    EVIL_LOSE_PAYOFF * (1.0 - correct_prob)
                );
            } else {
                node->counterfactual_values[player](viewpoint) += counterfactual_reach_prob * (
                    GOOD_LOSE_PAYOFF * correct_prob +
                    GOOD_WIN_PAYOFF * (1.0 - correct_prob)
                );
            }
        }

        double assassin_counterfactual_reach_prob = reach_prob / node->reach_probs[assassin](assassin_viewpoint);
        double expected_assassin_payoff = assassin_counterfactual_reach_prob * (
            EVIL_WIN_PAYOFF * correct_prob +
            EVIL_LOSE_PAYOFF * (1.0 - correct_prob)
        );
        for (int assassin_choice = 0; assassin_choice < NUM_PLAYERS; assassin_choice++) {
            // double choice_prob = node->merlin_strategy->at(assassin)(assassin_viewpoint, assassin_choice);
            double payoff = (
                (assassin_choice == merlin) ?
                (EVIL_WIN_PAYOFF * assassin_counterfactual_reach_prob) :
                (EVIL_LOSE_PAYOFF * assassin_counterfactual_reach_prob)
            );

            node->merlin_regrets->at(assassin)(assassin_viewpoint, assassin_choice) += payoff - expected_assassin_payoff;
        }
    }

    #ifdef CFR_PLUS
    for (int player = 0; player < NUM_PLAYERS; player++) {
        node->merlin_regrets->at(player) = node->merlin_regrets->at(player).max(0.0);
    }
    #endif
};

static void calculate_terminal_cfvs(LookaheadNode* node, const AssignmentProbs& starting_probs) {
    for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
        int evil = ASSIGNMENT_TO_EVIL[i];
        double reach_prob = (*(node->full_reach_probs))(i) * starting_probs(i);
        for (int player = 0; player < NUM_PLAYERS; player++) {
            int viewpoint = ASSIGNMENT_TO_VIEWPOINT[i][player];
            double counterfactual_reach_prob = reach_prob / node->reach_probs[player](viewpoint);
            if ((1 << player) & evil) {
                node->counterfactual_values[player](viewpoint) += counterfactual_reach_prob * EVIL_WIN_PAYOFF; // In these terminal nodes, evil wins, good loses.
            } else {
                node->counterfactual_values[player](viewpoint) += counterfactual_reach_prob * GOOD_LOSE_PAYOFF;
            }
        }
    }
}

static void calculate_neural_net_cfvs(LookaheadNode* node, const AssignmentProbs& starting_probs) {
    AssignmentProbs real_probs = (*(node->full_reach_probs)) * starting_probs;
    double sum = real_probs.sum();
    if (sum == 0.0) {
        return;
    }

    AssignmentProbs normalized_probs = real_probs / sum;

    node->nn_model->predict(node->proposer, normalized_probs, node->counterfactual_values);

    for (int i = 0; i < NUM_PLAYERS; i++) {
        // Re-normalize the values so they are counterfactual.
        node->counterfactual_values[i] /= node->reach_probs[i];
        node->counterfactual_values[i] *= sum;
    }
};

void calculate_counterfactual_values(LookaheadNode* node, const AssignmentProbs& starting_probs) {
    for (auto& child : node->children) {
        calculate_counterfactual_values(child.get(), starting_probs);
    }

    for (int player = 0; player < NUM_PLAYERS; player++) {
        node->counterfactual_values[player].setZero();
    }

    switch (node->type) {
    case PROPOSE:
        calculate_propose_cfvs(node); break;
    case VOTE:
        calculate_vote_cfvs(node); break;
    case MISSION:
        calculate_mission_cfvs(node); break;
    case TERMINAL_MERLIN:
        calculate_merlin_cfvs(node, starting_probs); break;
    case TERMINAL_NO_CONSENSUS:
    case TERMINAL_TOO_MANY_FAILS:
        calculate_terminal_cfvs(node, starting_probs); break;
    case TERMINAL_PROPOSE_NN:
        calculate_neural_net_cfvs(node, starting_probs); break;
    }

    #ifndef NDEBUG
    double check = 0.0;
    for (int i = 0; i < NUM_PLAYERS; i++) {
        check += (node->counterfactual_values[i] * node->reach_probs[i]).sum();
    }
    if (abs(check) > 1e-5) {
        std::cout << "NODE TYPE: " << node->typeAsString() << std::endl;
        std::cout << "MY SUM: " << check << std::endl;
        for (const auto& child : node->children) {
            double s = 0.0;
            for (int i = 0; i < NUM_PLAYERS; i++) s += (child->counterfactual_values[i] * child->reach_probs[i]).sum();
            std::cout << "CHILD SUM: " << s << std::endl;
        }
    }
    ASSERT(abs(check), <, 1e-5);
    assert(abs(check) < 1e-5);
    #endif
}

void cfr_get_values(
    LookaheadNode* root,
    const int iterations,
    const int wait_iterations,
    const AssignmentProbs& starting_probs,
    const bool save_strategy,
    ViewpointVector* values
) {
    ViewpointVector last_values[NUM_PLAYERS];
    for (int i = 0; i < NUM_PLAYERS; i++) {
        values[i].setZero();
        last_values[i].setZero();
    }

    long long total_size = 0;
    for (int iter = 0; iter < iterations; iter++) {
        int weight = (iter < wait_iterations) ? 0 : (iter - wait_iterations);

        calculate_strategy(root, (save_strategy) ? weight : 0.0);
        calculate_counterfactual_values(root, starting_probs);

        bool equals_previous_iteration = !save_strategy; // Save strategy disables checking for early finish
        for (int i = 0; i < NUM_PLAYERS; i++) {
            if (!last_values[i].isApprox(root->counterfactual_values[i])) {
                equals_previous_iteration = false;
            }
        }
        if (equals_previous_iteration) {
            for (int i = 0; i < NUM_PLAYERS; i++) {
                values[i] = root->counterfactual_values[i];
            }
            return;
        }

        total_size += weight;
        for (int player = 0; player < NUM_PLAYERS; player++) {
            last_values[player] = root->counterfactual_values[player];
            values[player] += root->counterfactual_values[player] * weight;
        }
    }

    for (int i = 0; i < NUM_PLAYERS; i++) {
        values[i] /= total_size;
    }
}

void calculate_cumulative_strategy(LookaheadNode* node) {
    switch (node->type) {
    case PROPOSE: {
        // Initialize the node's memory
        auto& player_cumulative = *(node->propose_cum);
        auto& player_strategy = *(node->propose_strategy);

        player_strategy = player_cumulative;

        ProposeData tmp_holder = player_strategy.colwise() / player_strategy.rowwise().sum();
        tmp_holder = tmp_holder.unaryExpr([](double v) { return std::isfinite(v) ? v : 1.0/NUM_PROPOSAL_OPTIONS; });
        player_strategy = tmp_holder; // (1.0 - TREMBLE_VALUE) * tmp_holder + TREMBLE_VALUE * ProposeData::Constant(1.0/NUM_PROPOSAL_OPTIONS);
    } break;
    case VOTE: {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            auto& player_cumulative = node->vote_cum->at(i);
            auto& player_strategy = node->vote_strategy->at(i);

            player_strategy = player_cumulative;

            VoteData tmp_holder = player_strategy.colwise() / player_strategy.rowwise().sum();
            tmp_holder = tmp_holder.unaryExpr([](double v) { return std::isfinite(v) ? v : 0.5; });
            player_strategy = tmp_holder; // (1.0 - TREMBLE_VALUE) * tmp_holder + TREMBLE_VALUE * VoteData::Constant(0.5);
        }
    } break;
    case MISSION: {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            if (((1 << i) & node->proposal) == 0) continue;
            auto& player_cumulative = node->mission_cum->at(i);
            auto& player_strategy = node->mission_strategy->at(i);

            player_strategy = player_cumulative;

            MissionData tmp_holder = player_strategy.colwise() / player_strategy.rowwise().sum();
            tmp_holder = tmp_holder.unaryExpr([](double v) { return std::isfinite(v) ? v : 0.5; });
            player_strategy = tmp_holder; // (1.0 - TREMBLE_VALUE) * tmp_holder + TREMBLE_VALUE * MissionData::Constant(0.5);
        }
    } break;
    case TERMINAL_MERLIN: {
        for (int i = 0; i < NUM_PLAYERS; i++) {
            auto& player_cumulative = node->merlin_cum->at(i);
            auto& player_strategy = node->merlin_strategy->at(i);

            player_strategy = player_cumulative;

            MerlinData tmp_holder = player_strategy.colwise() / player_strategy.rowwise().sum();
            tmp_holder = tmp_holder.unaryExpr([](double v) { return std::isfinite(v) ? v : 1.0/NUM_PLAYERS; });
            player_strategy = tmp_holder; // (1.0 - TREMBLE_VALUE) * tmp_holder + TREMBLE_VALUE * MerlinData::Constant(1.0/NUM_PLAYERS);
        }
        // Intentional missing break.
    }
    case TERMINAL_NO_CONSENSUS:
    case TERMINAL_TOO_MANY_FAILS:
    case TERMINAL_PROPOSE_NN: {
        populate_full_reach_probs(node);
    } break;
    default: break;
    }

    fill_reach_probabilities(node);

    for (auto& child : node->children) {
        calculate_cumulative_strategy(child.get());
    }
}
