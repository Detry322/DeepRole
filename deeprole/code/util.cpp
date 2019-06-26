#include <vector>
#include <algorithm>
#include <sstream>
#include <map>

#include "util.h"
#include "cfr_plus.h"

std::mt19937 rng;

void seed_rng() {
    std::random_device rd;
    std::array<int, std::mt19937::state_size> seed_data;
    std::generate_n(seed_data.data(), seed_data.size(), std::ref(rd));
    std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
    rng = std::mt19937(seq);
}

std::string Initialization::Stringify() const {
    static Eigen::IOFormat CSVFmt(10, Eigen::DontAlignCols, ",", ",", "", "", "", "");
    std::stringstream stream;
    // Add starting data
    stream << depth << ",";
    stream << num_succeeds << ",";
    stream << num_fails << ",";
    stream << propose_count << ",";
    stream << proposer << ",";
    stream << iterations << ",";
    stream << wait_iterations << ",";
    stream << generate_start_technique << ",";
    stream << starting_probs.format(CSVFmt) << ",";
    for (int i = 0; i < NUM_PLAYERS; i++) {
        stream << solution_values[i].format(CSVFmt) << ((i < NUM_PLAYERS - 1) ? "," : "");
    }
    return stream.str();
}

void fill_random_probability(const double amt_left, const int length, double* probs) {
    if (length == 1) {
        *probs = amt_left;
        return;
    }

    std::uniform_real_distribution<> dis(0.0, amt_left);
    double part_a = dis(rng);
    double part_b = amt_left - part_a;

    fill_random_probability(part_a, length/2, probs);
    fill_random_probability(part_b, length - (length/2), probs + (length/2));
}

void generate_merlin_probs(double* merlin_probs) {
    double tmp_merlin[NUM_PLAYERS];
    std::vector<int> player_to_index;
    for (int i = 0; i < NUM_PLAYERS; i++) {
        player_to_index.push_back(i);
    }
    std::shuffle(player_to_index.begin(), player_to_index.end(), rng);
    fill_random_probability(1.0, NUM_PLAYERS, tmp_merlin);

    for (int i = 0; i < NUM_PLAYERS; i++) {
        merlin_probs[player_to_index[i]] = tmp_merlin[i];
    }
}

std::vector<int> get_random_fail_rounds(const int num_succeeds, const int num_fails) {
    std::vector<int> result;
    for (int i = 0; i < (num_succeeds + num_fails); i++) {
        result.push_back(ROUND_TO_PROPOSE_SIZE[i]);
    }
    std::shuffle(result.begin(), result.end(), rng);
    result.resize(num_fails);
    return result;
}

std::vector<uint32_t> get_random_evil_possibilities(const int num_succeeds, const int num_fails) {
    std::vector<int> fail_rounds = get_random_fail_rounds(num_succeeds, num_fails);

    std::uniform_int_distribution<> get_num_fails(1, 2);
    std::uniform_int_distribution<> get_proposal(0, NUM_PROPOSAL_OPTIONS - 1);

    std::vector<std::pair<uint32_t, int>> fails;
    for (int round_size : fail_rounds) {
        int num_fails = get_num_fails(rng);
        const int* index_to_proposal = (round_size == 2) ? INDEX_TO_PROPOSAL_2 : INDEX_TO_PROPOSAL_3;
        uint32_t proposal = index_to_proposal[get_proposal(rng)];
        fails.push_back(std::make_pair(proposal, num_fails));
    }

    std::vector<uint32_t> evil_possibilities;

    for (int i = 0; i < NUM_PROPOSAL_OPTIONS; i++) {
        uint32_t evil = INDEX_TO_PROPOSAL_2[i];
        bool valid = true;

        for (auto fail : fails) {
            if (__builtin_popcount(fail.first & evil) < fail.second) {
                valid = false;
                break;
            }
        }

        if (valid) {
            evil_possibilities.push_back(evil);
        }
    }

    if (evil_possibilities.size() == 0) {
        // If it's impossible, try again
        return get_random_evil_possibilities(num_succeeds, num_fails);
    } else {
        return evil_possibilities;
    }
}

std::map<std::pair<int, int>, double> generate_evil_probs(const int num_succeeds, const int num_fails) {
    auto evil_possibilities = get_random_evil_possibilities(num_succeeds, num_fails);

    std::shuffle(evil_possibilities.begin(), evil_possibilities.end(), rng);
    double evil_probs[evil_possibilities.size()];
    fill_random_probability(1.0, evil_possibilities.size(), evil_probs);

    std::map<std::pair<int, int>, double> result;

    std::uniform_real_distribution<> get_split(0.0, 1.0);

    for (size_t i = 0; i < evil_possibilities.size(); i++) {
        uint32_t evil = evil_possibilities[i];
        double prob = evil_probs[i];
        int player_1 = __builtin_ctz(evil);
        evil &= ~(1 << player_1);
        int player_2 = __builtin_ctz(evil);

        double split_prob = get_split(rng);

        result[std::make_pair(player_1, player_2)] = prob * split_prob;
        result[std::make_pair(player_2, player_1)] = prob * (1.0 - split_prob);
    }

    return result;
}

void generate_starting_probs_v1(AssignmentProbs& starting_probs, const int num_succeeds, const int num_fails) {
    double merlin_probs[NUM_PLAYERS];
    generate_merlin_probs(merlin_probs);
    auto evil_probs = generate_evil_probs(num_succeeds, num_fails);

    starting_probs.setZero();
    starting_probs = AssignmentProbs::Constant(1.0/NUM_ASSIGNMENTS);

    for (int i = 0; i < NUM_ASSIGNMENTS; i++) {
        int merlin = ASSIGNMENT_TO_ROLES[i][0];
        int assassin = ASSIGNMENT_TO_ROLES[i][1];
        int minion = ASSIGNMENT_TO_ROLES[i][2];
        starting_probs(i) = merlin_probs[merlin] * evil_probs[std::make_pair(assassin, minion)];
    }
    double sum = starting_probs.sum();

    if (sum == 0.0) {
        // Try again, something went wrong.
        generate_starting_probs_v1(starting_probs, num_succeeds, num_fails);
        return;
    } else {
        starting_probs /= sum;
    }
}

void prepare_initialization(
    const int depth,
    const int num_succeeds,
    const int num_fails,
    const int propose_count,
    Initialization* init
) {
    std::uniform_int_distribution<> dis(0, NUM_PLAYERS - 1);

    init->depth = depth;
    init->num_succeeds = num_succeeds;
    init->num_fails = num_fails;
    init->proposer = dis(rng);
    init->propose_count = propose_count;
    init->generate_start_technique = "v1";
    generate_starting_probs_v1(init->starting_probs, num_succeeds, num_fails);
}

void run_initialization_with_cfr(
    const int iterations,
    const int wait_iterations,
    const std::string& model_search_dir,
    Initialization* init
) {
    init->iterations = iterations;
    init->wait_iterations = wait_iterations;

    auto lookahead = create_avalon_lookahead(
        init->num_succeeds,
        init->num_fails,
        init->proposer,
        init->propose_count,
        init->depth,
        model_search_dir
    );

    AssignmentProbs& starting_probs = init->starting_probs;
    ViewpointVector* out_values = init->solution_values;
    cfr_get_values(lookahead.get(), iterations, wait_iterations, starting_probs, true, out_values);
}
