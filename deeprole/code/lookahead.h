#ifndef LOOKAHEAD_H_
#define LOOKAHEAD_H_

#include <memory>
#include <array>
#include <vector>

#include "lookup_tables.h"
#include "eigen_types.h"

#include "nn.h"

enum NodeType {
    PROPOSE,
    VOTE,
    MISSION,
    TERMINAL_MERLIN,
    TERMINAL_NO_CONSENSUS,
    TERMINAL_TOO_MANY_FAILS,
    TERMINAL_PROPOSE_NN
};

struct LookaheadNode {
    NodeType type;
    int num_succeeds;
    int num_fails;
    int proposer;
    int propose_count;
    uint32_t proposal;
    int merlin_pick;

    ViewpointVector reach_probs[NUM_PLAYERS];
    ViewpointVector counterfactual_values[NUM_PLAYERS];
    
    std::unique_ptr<AssignmentProbs> full_reach_probs;

    std::unique_ptr<ProposeData> propose_regrets;
    std::unique_ptr<ProposeData> propose_strategy;
    std::unique_ptr<ProposeData> propose_cum;

    std::unique_ptr<std::array<VoteData, NUM_PLAYERS>> vote_regrets;
    std::unique_ptr<std::array<VoteData, NUM_PLAYERS>> vote_strategy;
    std::unique_ptr<std::array<VoteData, NUM_PLAYERS>> vote_cum;

    std::unique_ptr<std::array<MissionData, NUM_PLAYERS>> mission_regrets;
    std::unique_ptr<std::array<MissionData, NUM_PLAYERS>> mission_strategy;
    std::unique_ptr<std::array<MissionData, NUM_PLAYERS>> mission_cum;

    std::unique_ptr<std::array<MerlinData, NUM_PLAYERS>> merlin_regrets;
    std::unique_ptr<std::array<MerlinData, NUM_PLAYERS>> merlin_strategy;
    std::unique_ptr<std::array<MerlinData, NUM_PLAYERS>> merlin_cum;

    std::vector<std::pair<uint32_t, int>> fails;
    std::vector<std::unique_ptr<LookaheadNode>> children;

    std::shared_ptr<Model> nn_model;

    LookaheadNode() = default;

    int round() const;

    std::string typeAsString() const;

    static std::unique_ptr<LookaheadNode> RootProposal(int num_succeeds, int num_fails, int proposer, int propose_count);
    static std::unique_ptr<LookaheadNode> CopyParent(const LookaheadNode& parent);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

std::unique_ptr<LookaheadNode> create_avalon_lookahead(
    const int num_succeeds,
    const int num_fails,
    const int proposer,
    const int propose_count,
    const int depth,
    const std::string& model_search_dir);

int count_lookahead_type(LookaheadNode* node, const NodeType type);
int count_lookahead(LookaheadNode* node);

#endif // LOOKAHEAD_H_
