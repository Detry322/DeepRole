#include "json.h"
#include <iomanip>

#include "serialization.h"

using json = nlohmann::json;
using namespace std;

void json_deserialize_starting_reach_probs(std::istream& in_stream, AssignmentProbs* starting_reach_probs) {
    cerr << "Deserializing. Enter a json array of numbers:" << endl;
    json j;
    in_stream >> j;
    std::vector<double> parsed_double_arr = j;

    if (parsed_double_arr.size() != NUM_ASSIGNMENTS) {
        throw std::length_error("Double array isn't the correct size");
    }

    for (int i = 0; i < parsed_double_arr.size(); i++) {
        (*starting_reach_probs)(i) = parsed_double_arr[i];
    }

    double reach_sum = starting_reach_probs->sum();

    if (abs(reach_sum - 1.0) >= 1e-9) {
        throw std::domain_error("Reach sum doesn't sum to 1.0");
    }

    *starting_reach_probs /= reach_sum;
}

void json_serialize_propose_node(const LookaheadNode* node, json& json_node) {
    json_node["succeeds"] = node->num_succeeds;
    json_node["fails"] = node->num_fails;
    json_node["proposer"] = node->proposer;
    json_node["propose_count"] = node->propose_count;

    json_node["propose_strat"] = eigen_to_double_vector(*(node->propose_strategy));
    json_node["propose_options"] = json::array();

    const int* proposal_options = (ROUND_TO_PROPOSE_SIZE[node->round()] == 2) ? INDEX_TO_PROPOSAL_2 : INDEX_TO_PROPOSAL_3;
    for (int i = 0; i < NUM_PROPOSAL_OPTIONS; i++) {
        json_node["propose_options"].push_back(proposal_options[i]);
    }
}

void json_serialize_vote_node(const LookaheadNode* node, json& json_node) {
    json_node["succeeds"] = node->num_succeeds;
    json_node["fails"] = node->num_fails;
    json_node["proposer"] = node->proposer;
    json_node["propose_count"] = node->propose_count;
    json_node["proposal"] = node->proposal;

    json_node["vote_strat"] = json::array();
    for (int i = 0; i < NUM_PLAYERS; i++) {
        json_node["vote_strat"].push_back(eigen_to_double_vector(node->vote_strategy->at(i)));
    }
}

void json_serialize_mission_node(const LookaheadNode* node, json& json_node) {
    json_node["succeeds"] = node->num_succeeds;
    json_node["fails"] = node->num_fails;
    json_node["proposal"] = node->proposal;

    json_node["mission_strat"] = json::array();
    for (int i = 0; i < NUM_PLAYERS; i++) {
        if (((1 << i) & node->proposal) == 0) {
            json_node["mission_strat"].push_back(nullptr);
        } else {
            json_node["mission_strat"].push_back(eigen_to_double_vector(node->mission_strategy->at(i)));
        }
    }
}

void json_serialize_merlin_node(const LookaheadNode* node, json& json_node) {
    json_node["succeeds"] = node->num_succeeds;
    json_node["fails"] = node->num_fails;

    json_node["merlin_strat"] = json::array();
    for (int i = 0; i < NUM_PLAYERS; i++) {
        json_node["merlin_strat"].push_back(eigen_to_double_vector(node->merlin_strategy->at(i)));
    }
}

void calc_nozero_belief(const LookaheadNode* node, const AssignmentProbs& starting_reach_probs, AssignmentProbs* nozero_belief) {
    for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
        for (int player = 0; player < NUM_PLAYERS; player++) {
            int viewpoint = ASSIGNMENT_TO_VIEWPOINT[i][player];
            (*nozero_belief)(i) *= node->reach_probs[player](viewpoint);
        }
    }
}

void json_serialize_nn_propose(const LookaheadNode* node, const AssignmentProbs& starting_reach_probs, json& json_node) {
    json_node["succeeds"] = node->num_succeeds;
    json_node["fails"] = node->num_fails;
    json_node["proposer"] = node->proposer;
    json_node["propose_count"] = node->propose_count;

    AssignmentProbs new_belief = *(node->full_reach_probs) * starting_reach_probs;
    AssignmentProbs nozero_belief = starting_reach_probs;
    calc_nozero_belief(node, starting_reach_probs, &nozero_belief);

    double sum = new_belief.sum();

    if (sum == 0.0) {
        json_node["new_belief"] = "impossible";
        json_node["nn_output"] = "n/a";
    } else {
        new_belief /= sum;
        nozero_belief /= nozero_belief.sum();
        ViewpointVector nn_output[NUM_PLAYERS];
        node->nn_model->predict(node->proposer, new_belief, nn_output);
        
        json_node["new_belief"] = eigen_to_single_vector(new_belief);
        json_node["nozero_belief"] = eigen_to_single_vector(nozero_belief);
        for (int i = 0; i < NUM_PLAYERS; i++) {
            json_node["nn_output"].push_back(eigen_to_single_vector(nn_output[i]));
        }
    }
}

void json_serialize_node(const LookaheadNode* node, const AssignmentProbs& starting_reach_probs, json& json_node) {
    json_node["type"] = node->typeAsString();

    switch (node->type) {
    case PROPOSE: {
        json_serialize_propose_node(node, json_node);
    } break;
    case VOTE: {
        json_serialize_vote_node(node, json_node);
    } break;
    case MISSION: {
        json_serialize_mission_node(node, json_node);
    } break;
    case TERMINAL_MERLIN: {
        json_serialize_merlin_node(node, json_node);
    } break;
    case TERMINAL_PROPOSE_NN: {
        json_serialize_nn_propose(node, starting_reach_probs, json_node);
    } break;
    case TERMINAL_NO_CONSENSUS:
    case TERMINAL_TOO_MANY_FAILS:
    default: break;
    }

    if (node->children.size() == 0) {
        return;
    }

    json children_nodes = json::array();

    for (const auto& child : node->children) {
        json child_json;
        json_serialize_node(child.get(), starting_reach_probs, child_json);
        children_nodes.push_back(child_json);
    }

    json_node["children"] = children_nodes;
};

void json_serialize_lookahead(const LookaheadNode* root, const AssignmentProbs& starting_reach_probs, std::ostream& out_stream) {
    json result;
    json_serialize_node(root, starting_reach_probs, result);
    out_stream << std::setprecision(17) << std::setw(2) << result << std::endl;
}
