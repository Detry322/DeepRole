#include <iostream>
#include <fstream>
#include <cstdint>
#include <cassert>
#include <random>
#include <mutex>
#include <thread>
#include <vector>
#include <cilk/cilk.h>

static thread_local uint32_t rngx=123456789, rngy=362436069, rngz=521288629;
uint32_t xorshf96(void) {          //period 2^96-1
    uint32_t t;
    rngx ^= rngx << 16;
    rngx ^= rngx >> 5;
    rngx ^= rngx << 1;

    t = rngx;
    rngx = rngy;
    rngy = rngz;
    rngz = t ^ rngx ^ rngy;

    return rngz;
}

float nextFloat() {
    float x;
    *((uint32_t*) &x) = (0x7fffff & xorshf96()) | 0x3f800000;
    return x - 1;
}

using namespace std;

float* proposal_regretsum = NULL;
float* proposal_stratsum = NULL;
float* voting_regretsum = NULL;
float* voting_stratsum = NULL;
float* mission_regretsum = NULL;
float* mission_stratsum = NULL;
float* merlin_regretsum = NULL;
float* merlin_stratsum = NULL;

const float BRANCHING_PROBABILITY = 0.1;
// 162 possible "missions failed with you on it"
// 21 possible missions failed with you proposed
// 5 possible rounds
const int HISTORY_SIZE = 162 * 21 * 5;
// 162 possible "missions failed with you on it"
// 243 possible "upvoted a failed mission"
const int MERLIN_HISTORY_SIZE = 162 * 243;
// 5 choose 3 and 5 choose 2 are the same.
const int NUM_POSSIBLE_PROPOSALS = 10;
// 30 merlin viewpoints (10 evil possibilities, 3 merlin) + 20 evil possibilities + 5 servant possibilities
const int NUM_EVIL_VIEWPOINTS = 20;
const int NUM_POSSIBLE_VIEWPOINTS = NUM_EVIL_VIEWPOINTS + 30 + 5;

const int NUM_PROPOSAL_BUCKETS = NUM_POSSIBLE_VIEWPOINTS * HISTORY_SIZE * 3;
const int NUM_VOTING_BUCKETS = NUM_POSSIBLE_VIEWPOINTS * HISTORY_SIZE * 3 * NUM_POSSIBLE_PROPOSALS;
const int NUM_MISSION_BUCKETS = NUM_EVIL_VIEWPOINTS * HISTORY_SIZE * NUM_POSSIBLE_PROPOSALS;
// 10 possible bad guy pairs
const int NUM_MERLIN_BUCKETS = 10 * MERLIN_HISTORY_SIZE;

const int NUM_PROPOSAL_FLOATS = NUM_PROPOSAL_BUCKETS * NUM_POSSIBLE_PROPOSALS;
const int NUM_VOTING_FLOATS = NUM_VOTING_BUCKETS * 2;
const int NUM_MISSION_FLOATS = NUM_MISSION_BUCKETS * 2;
const int NUM_MERLIN_FLOATS = NUM_MERLIN_BUCKETS * 5;


// Lookup tables for buckets.
const int PROPOSAL_FAIL_LOOKUP[243] = {0, 1, 2, 3, 4, -1, 5, -1, -1, 6, 7, -1, 8, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1, -1, -1, -1, -1, 10, 11, -1, 12, -1, -1, -1, -1, -1, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 16, -1, 17, -1, -1, -1, -1, -1, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 19, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 20, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
const int MISSION_FAIL_LOOKUP[243] = {0, -1, -1, -1, 1, -1, -1, -1, 2, -1, 3, -1, 4, 5, 6, -1, 7, 8, -1, -1, 9, -1, 10, 11, 12, 13, 14, -1, 15, -1, 16, 17, 18, -1, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, -1, 30, 31, 32, 33, 34, 35, 36, -1, -1, -1, 37, -1, 38, 39, 40, 41, 42, -1, 43, 44, 45, 46, 47, 48, 49, -1, 50, 51, 52, 53, 54, -1, 55, -1, -1, -1, 56, -1, 57, 58, 59, -1, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, -1, 71, 72, 73, 74, 75, 76, 77, -1, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, -1, 95, 96, 97, 98, 99, -1, 100, -1, -1, -1, 101, 102, 103, 104, 105, 106, 107, -1, 108, 109, 110, 111, 112, -1, 113, -1, -1, 114, 115, -1, 116, -1, -1, -1, -1, -1, -1, -1, 117, -1, 118, 119, 120, 121, 122, -1, 123, 124, 125, 126, 127, 128, 129, -1, 130, 131, 132, 133, 134, -1, 135, -1, -1, -1, 136, 137, 138, 139, 140, 141, 142, -1, 143, 144, 145, 146, 147, -1, 148, -1, -1, 149, 150, -1, 151, -1, -1, -1, -1, -1, 152, 153, 154, 155, 156, -1, 157, -1, -1, 158, 159, -1, 160, -1, -1, -1, -1, -1, 161, -1, -1, -1, -1, -1, -1, -1, -1};
// Keyed on (you, them)
const int EVIL_LOOKUP[25] = {-1, 0, 2, 4, 6, 1, -1, 8, 10, 12, 3, 9, -1, 14, 16, 5, 11, 15, -1, 18, 7, 13, 17, 19, -1};
// Keyed on (you, bad_index)
const int MERLIN_LOOKUP[50] = {-1, -1, -1, -1, 0, 1, 2, 3, 4, 5, -1, 6, 7, 8, -1, -1, -1, 9, 10, 11, 12, -1, 13, 14, -1, 15, 16, -1, -1, 17, 18, 19, -1, 20, 21, -1, 22, -1, 23, -1, 24, 25, 26, -1, 27, 28, -1, 29, -1, -1};
// Keyed on proposal bitstring
const int PROPOSAL_TO_INDEX_LOOKUP[32] = {-1, -1, -1, 0, -1, 1, 4, 0, -1, 2, 5, 1, 7, 3, 6, -1, -1, 3, 6, 2, 8, 4, 7, -1, 9, 5, 8, -1, 9, -1, -1, -1};
const int INDEX_TO_PROPOSAL_2[10] = {3, 5, 9, 17, 6, 10, 18, 12, 20, 24};
const int INDEX_TO_PROPOSAL_3[10] = {7, 11, 19, 13, 21, 25, 14, 22, 26, 28};
const int ROUND_TO_PROPOSE_SIZE[5] = {2, 3, 2, 3, 3};

const int NUM_MUTEXES = 128;

mutex mtxs[NUM_MUTEXES];

int games_explored = 0;

enum Status {
    Status_propose,
    Status_vote,
    Status_mission,
    Status_merlin
};

class RoundState {
public:
    Status status;
    uint32_t proposal;
    uint8_t proposer;
    uint8_t propose_count;
    uint8_t succeeds;
    uint8_t fails;
    uint8_t missions_with_fail[5];
    uint8_t proposals_with_fail[5];
    uint8_t upvote_fail[5];
    uint32_t votes;

    bool is_terminal() const {
        return (this->propose_count > 4) || (this->fails > 2) || (this->succeeds > 2);
    }

    float terminal_payoff(int player, uint32_t hidden_state) const {
        assert(this->is_terminal());
        bool is_evil = ((hidden_state & 0xFF) == player) || (((hidden_state >> 8) & 0xFF) == player);
        if (is_evil) {
            return (this->succeeds > 2) ? -1.0 : 1.0;
        } else {
            return (this->succeeds > 2) ? 1.0 : -1.0;
        }
    }

    int round() const {
        return this->succeeds + this->fails;
    }

    int get_history_bucket() const {
        int proposals_with_fail_big_index = (
            1  * this->proposals_with_fail[0] +
            3  * this->proposals_with_fail[1] +
            9  * this->proposals_with_fail[2] +
            27 * this->proposals_with_fail[3] +
            81 * this->proposals_with_fail[4]
        );
        int missions_with_fail_big_index = (
            1  * this->missions_with_fail[0] +
            3  * this->missions_with_fail[1] +
            9  * this->missions_with_fail[2] +
            27 * this->missions_with_fail[3] +
            81 * this->missions_with_fail[4]
        );
        int proposals_with_fail_index = PROPOSAL_FAIL_LOOKUP[proposals_with_fail_big_index];
        int missions_with_fail_index = MISSION_FAIL_LOOKUP[missions_with_fail_big_index];
        assert(proposals_with_fail_index >= 0);
        assert(missions_with_fail_index >= 0);

        int result = (162 * 5 * proposals_with_fail_index) + (5 * missions_with_fail_index) + this->round();
        assert(result >= 0 && result < HISTORY_SIZE);
        return result;
    }

    int get_merlin_history_bucket() const {
        int missions_with_fail_big_index = (
            1  * this->missions_with_fail[0] +
            3  * this->missions_with_fail[1] +
            9  * this->missions_with_fail[2] +
            27 * this->missions_with_fail[3] +
            81 * this->missions_with_fail[4]
        );
        int missions_with_fail_index = MISSION_FAIL_LOOKUP[missions_with_fail_big_index];
        assert(missions_with_fail_index >= 0);
        int upvoted_fail = (
            1  * this->upvote_fail[0] +
            3  * this->upvote_fail[1] +
            9  * this->upvote_fail[2] +
            27 * this->upvote_fail[3] +
            81 * this->upvote_fail[4]
        );
        int index = (243 * missions_with_fail_index) + upvoted_fail;
        assert(index >= 0 && index < MERLIN_HISTORY_SIZE);
        return index;
    }

    static void ProgressProposal(const RoundState& old_state, uint32_t proposal, RoundState* new_state) {
        *new_state = old_state;
        new_state->proposal = proposal;
        new_state->status = Status_vote;
    }

    static void ProgressVote(const RoundState& old_state, uint32_t votes, RoundState* new_state) {
        *new_state = old_state;
        if (__builtin_popcount(votes) >= 3) {
            // Vote passed
            new_state->status = Status_mission;
            new_state->votes = votes;
        } else {
            // Vote failed
            new_state->propose_count++;
            new_state->proposer = (new_state->proposer + 1) % 5;
            new_state->status = Status_propose;
        }
    }

    static void ProgressMission(const RoundState& old_state, bool did_fail, RoundState* new_state) {
        *new_state = old_state;
        new_state->proposer = (new_state->proposer + 1) % 5;
        new_state->propose_count = 0;
        new_state->status = Status_propose;
        if (did_fail) {
            new_state->fails++;
            new_state->proposals_with_fail[old_state.proposer]++;
            for (int i = 0; i < 5; i++) {
                new_state->upvote_fail[i] += 0x1 & (old_state.votes >> i);
                new_state->missions_with_fail[i] += 0x1 & (old_state.proposal >> i);
            }
        } else {
            if (old_state.succeeds == 2) {
                new_state->status = Status_merlin;
            } else {
                new_state->succeeds++;
            }
        }
    }

    static void ProgressMerlin(const RoundState& old_state, bool picked_correctly, RoundState* new_state) {
        *new_state = old_state;
        if (picked_correctly) {
            new_state->fails = 5; // Sentinel value
        } else {
            new_state->succeeds++;
        }
    }
};

float mccfr_propose(int t, const RoundState& state, uint32_t hidden_state, int traverser, float pi);
float mccfr_vote(int t, const RoundState& state, uint32_t hidden_state, int traverser, float pi);
float mccfr_mission(int t, const RoundState& state, uint32_t hidden_state, int traverser, float pi);
float mccfr_merlin(int t, const RoundState& state, uint32_t hidden_state, int traverser, float pi);
float mccfr(int t, const RoundState& state, uint32_t hidden_state, int traverser, float pi);

int get_general_perspective(int player, uint32_t hidden_state) {
    if (((hidden_state >> 16) & 0xFF) == player) {
        // If we're merlin
        int bad_guy_index = EVIL_LOOKUP[5*((hidden_state >> 8) & 0xFF) + (hidden_state & 0xFF)];
        assert(bad_guy_index >= 0);
        int merlin_perspective = MERLIN_LOOKUP[10 * player + (bad_guy_index / 2)];
        assert(merlin_perspective >= 0);
        assert(merlin_perspective + 25 < NUM_POSSIBLE_VIEWPOINTS);
        return merlin_perspective + 25;
    } else if (player == (hidden_state & 0xFF) || player == ((hidden_state >> 8) & 0xFF)) {
        int partner = ((hidden_state & 0xFF) == player) ? ((hidden_state >> 8) & 0xFF) : (hidden_state & 0xFF);
        int perspective = EVIL_LOOKUP[5*player + partner];
        assert(perspective >= 0);
        return perspective + 5;
    } else {
        return player;
    }
}

int get_propose_bucket(const RoundState& state, int player, uint32_t hidden_state) {
    assert(state.status == Status_propose);
    assert(state.proposer == player);
    int perspective = get_general_perspective(player, hidden_state);
    int propose_num = state.propose_count / 2;
    int history = state.get_history_bucket();
    int index = (NUM_POSSIBLE_VIEWPOINTS * 3 * history) + (3 * perspective) + propose_num;
    assert(index >= 0 && index < NUM_PROPOSAL_BUCKETS);
    return index;
}

int get_vote_bucket(const RoundState& state, int player, uint32_t hidden_state) {
    assert(state.status == Status_vote);
    int perspective = get_general_perspective(player, hidden_state);
    int propose_num = state.propose_count / 2;
    int history = state.get_history_bucket();
    int proposal = PROPOSAL_TO_INDEX_LOOKUP[state.proposal];
    assert(proposal >= 0);
    int index = (
        (NUM_POSSIBLE_VIEWPOINTS * 3 * NUM_POSSIBLE_PROPOSALS * history) +
        (3 * NUM_POSSIBLE_PROPOSALS * perspective) +
        (NUM_POSSIBLE_PROPOSALS * propose_num) +
        proposal
    );
    assert(index >= 0 && index < NUM_VOTING_BUCKETS);
    return index;
}

int get_mission_bucket(const RoundState& state, int player, uint32_t hidden_state) {
    // Check if it's the mission status.
    assert(state.status == Status_mission);
    // Check if player is on the mission
    assert(((1 << player) & state.proposal) != 0);
    // Check if player is evil
    assert(player == (hidden_state & 0xFF) || player == ((hidden_state >> 8) & 0xFF));
    int partner = ((hidden_state & 0xFF) == player) ? ((hidden_state >> 8) & 0xFF) : (hidden_state & 0xFF);

    int perspective = EVIL_LOOKUP[5*player + partner];
    assert(perspective >= 0);
    int history = state.get_history_bucket();
    int proposal = PROPOSAL_TO_INDEX_LOOKUP[state.proposal];
    assert(proposal >= 0);

    int index = (NUM_POSSIBLE_PROPOSALS * HISTORY_SIZE * perspective) + (NUM_POSSIBLE_PROPOSALS * history) + proposal;
    assert(index >= 0 && index < NUM_MISSION_BUCKETS);
    return index;
}

int get_merlin_bucket(const RoundState& state, int player, uint32_t hidden_state) {
    // Check if it's the mission status.
    assert(state.status == Status_merlin);
    // Check if player is assassin
    assert(player == ((hidden_state >> 8) & 0xFF));
    int perspective = EVIL_LOOKUP[5*(hidden_state & 0xFF) + player] / 2;
    int history = state.get_merlin_history_bucket();
    int index = (MERLIN_HISTORY_SIZE * perspective) + history;
    assert(index >= 0 && index < NUM_MERLIN_BUCKETS);
    return index;
}

default_random_engine generator;
uint32_t random_hidden() {
    uniform_int_distribution<int> p(0, 4);
    int merlin = xorshf96() % 5;
    int assassin = xorshf96() % 5;
    while (assassin == merlin) {
        assassin = xorshf96() % 5;
    }
    int minion = xorshf96() % 5;
    while (minion == merlin || minion == assassin) {
        minion = xorshf96() % 5;
    }
    return (merlin << 16) | (assassin << 8) | (minion << 0);
}

void calculate_strategy(float* regretsum, int num_actions, float* strategy_out) {
    float total = 0;
    for (int i = 0; i < num_actions; i++) {
        strategy_out[i] = (regretsum[i] > 0) ? regretsum[i] : 0;
        total += strategy_out[i];
    }
    if (total == 0) {
        for (int i = 0; i < num_actions; i++) {
            strategy_out[i] = 1.0 / num_actions;
        }
        return;
    }
    for (int i = 0; i < num_actions; i++) {
        strategy_out[i] /= total;
    }
}

int get_move(float* strategy, int num_actions) {
    float target = nextFloat();
    float running_total = 0.0;
    int i = 0;
    for (; i < num_actions - 1; i++) {
        running_total += strategy[i];
        if (running_total > target) {
            break;
        }
    }
    return i;
}

float mccfr_propose(int t, const RoundState& state, uint32_t hidden_state, int traverser, float pi) {
    assert(state.status == Status_propose);
    RoundState new_state = {};

    int bucket = get_propose_bucket(state, state.proposer, hidden_state);
    float strategy[NUM_POSSIBLE_PROPOSALS];
    calculate_strategy(proposal_regretsum + NUM_POSSIBLE_PROPOSALS * bucket, NUM_POSSIBLE_PROPOSALS, strategy);

    const int* proposal_map = (state.round() == 0 || state.round() == 2) ? INDEX_TO_PROPOSAL_2 : INDEX_TO_PROPOSAL_3;
    if (state.proposer != traverser) {
        int move = get_move(strategy, NUM_POSSIBLE_PROPOSALS);
        RoundState::ProgressProposal(state, proposal_map[move], &new_state);
        return mccfr(t, new_state, hidden_state, traverser, pi);
    }

    // cout << "Making proposal decision" << endl;
    float action_values[NUM_POSSIBLE_PROPOSALS];
    float value = 0;
    RoundState new_states[NUM_POSSIBLE_PROPOSALS] = {};
    cilk_for (int i = 0; i < NUM_POSSIBLE_PROPOSALS; i++) {
        RoundState::ProgressProposal(state, proposal_map[i], &new_states[i]);
        action_values[i] = mccfr(t, new_states[i], hidden_state, traverser, pi * strategy[i]);
    }
    for (int i = 0; i < NUM_POSSIBLE_PROPOSALS; i++) {
        value += action_values[i] * strategy[i];
    }

    mtxs[bucket % NUM_MUTEXES].lock();
    for (int i = 0; i < NUM_POSSIBLE_PROPOSALS; i++) {
        proposal_regretsum[bucket * NUM_POSSIBLE_PROPOSALS + i] += (action_values[i] - value) * t;
        proposal_stratsum[bucket * NUM_POSSIBLE_PROPOSALS + i] += strategy[i] * pi * t;
        assert(proposal_stratsum[bucket * NUM_POSSIBLE_PROPOSALS + i] >= 0);
    }
    mtxs[bucket % NUM_MUTEXES].unlock();

    return value;
}

float mccfr_vote(int t, const RoundState& state, uint32_t hidden_state, int traverser, float pi) {
    assert(state.status == Status_vote);
    RoundState new_state = {};

    uint32_t votes = 0;
    float strategy[2];
    for (int i = 0; i < 5; i++) {
        if (i == traverser) {
            continue;
        }

        int bucket = get_vote_bucket(state, i, hidden_state);
        calculate_strategy(voting_regretsum + 2 * bucket, 2, strategy);
        int move = get_move(strategy, 2);
        votes |= move << i;
    }

    // Can do an optimization here if no choice will change outcome. (i.e. the mission has at least 3 upvotes and there are no bad people on it)
    // if my vote won't matter
    int mybucket = get_vote_bucket(state, traverser, hidden_state);

    uint32_t evil_bitmap = (1 << (hidden_state & 0xFF)) | (1 << ((hidden_state >> 8) & 0xFF));
    if ((__builtin_popcount(votes) < 2) || (__builtin_popcount(votes) > 2 && (evil_bitmap & state.proposal) == 0)) {
        RoundState::ProgressVote(state, votes, &new_state);
        for (int i = 0; i < 2; i++) {
            voting_stratsum[mybucket * 2 + i] += strategy[i] * pi * t;
        }
        return mccfr(t, new_state, hidden_state, traverser, pi);
    }

    calculate_strategy(voting_regretsum + 2 * mybucket, 2, strategy);

    if (state.propose_count < 4 && nextFloat() > BRANCHING_PROBABILITY) {
        int move = get_move(strategy, 2);
        votes |= move << traverser;
        RoundState::ProgressVote(state, votes, &new_state);
        return mccfr(t, new_state, hidden_state, traverser, pi);
    }

    // cout << "Making voting decision" << endl;
    float action_values[2];
    float value = 0;
    RoundState new_states[2] = {};
    cilk_for (int i = 0; i < 2; i++) {
        RoundState::ProgressVote(state, votes | (i << traverser), &new_states[i]);
        action_values[i] = mccfr(t, new_states[i], hidden_state, traverser, pi * strategy[i]);
    }
    for (int i = 0; i < 2; i++) {
        value += action_values[i] * strategy[i];
    }

    mtxs[mybucket % NUM_MUTEXES].lock();
    for (int i = 0; i < 2; i++) {
        voting_regretsum[mybucket * 2 + i] += (action_values[i] - value) * t;
        voting_stratsum[mybucket * 2 + i] += strategy[i] * pi * t;
        assert(voting_stratsum[mybucket * 2 + i] >= 0);
    }
    mtxs[mybucket % NUM_MUTEXES].unlock();

    return value;
}

float mccfr_mission(int t, const RoundState& state, uint32_t hidden_state, int traverser, float pi) {
    assert(state.status == Status_mission);
    RoundState new_state = {};

    uint32_t evil_bitmap = (1 << (hidden_state & 0xFF)) | (1 << ((hidden_state >> 8) & 0xFF));
    if ((evil_bitmap & state.proposal) == 0) {
        RoundState::ProgressMission(state, false, &new_state);
        return mccfr(t, new_state, hidden_state, traverser, pi);
    }

    float strategy[2];
    bool did_fail = false;
    for (int i = 0; i < 5; i++) {
        // If the examined person is me.
        if (i == traverser) continue;
        // If the person isn't on the mission.
        if (((1 << i) & state.proposal) == 0) continue;
        // If the person isn't evil.
        if (((1 << i) & evil_bitmap) == 0) continue;

        int bucket = get_mission_bucket(state, i, hidden_state);
        calculate_strategy(mission_regretsum + 2*bucket, 2, strategy);
        int fail = get_move(strategy, 2);
        did_fail |= (bool) fail;
        if (did_fail) break;
    }

    // If I'm not on the mission, we're done. If I'm not evil, we're done.
    if (((1 << traverser) & state.proposal) == 0 || ((1 << traverser) & evil_bitmap) == 0) {
        RoundState::ProgressMission(state, did_fail, &new_state);
        return mccfr(t, new_state, hidden_state, traverser, pi);
    }

    // An optimization can be done here to remove instances where my choice doesn't matter.
    int mybucket = get_mission_bucket(state, traverser, hidden_state);
    calculate_strategy(mission_regretsum + 2 * mybucket, 2, strategy);

    if (did_fail) {
        RoundState::ProgressMission(state, did_fail, &new_state);
        for (int i = 0; i < 2; i++) {
            mission_stratsum[mybucket * 2 + i] += pi * strategy[i] * t;
        }
        return mccfr(t, new_state, hidden_state, traverser, pi);
    }

    // cout << "Making mission decision" << endl;
    float action_values[2];
    float value = 0;
    RoundState new_states[2] = {};
    cilk_for (int i = 0; i < 2; i++) {
        RoundState::ProgressMission(state, did_fail | (bool) i, &new_states[i]);
        action_values[i] = mccfr(t, new_states[i], hidden_state, traverser, pi * strategy[i]);
    }
    for (int i = 0; i < 2; i++) {
        value += action_values[i] * strategy[i];
    }

    mtxs[mybucket % NUM_MUTEXES].lock();
    for (int i = 0; i < 2; i++) {
        mission_regretsum[mybucket * 2 + i] += (action_values[i] - value) * t;
        mission_stratsum[mybucket * 2 + i] += pi * strategy[i] * t;
    }
    mtxs[mybucket % NUM_MUTEXES].unlock();
    
    return value;
}

float mccfr_merlin(int t, const RoundState& state, uint32_t hidden_state, int traverser, float pi) {
    assert(state.status == Status_merlin);
    RoundState new_state = {};

    int assassin = (hidden_state >> 8) & 0xFF;
    int merlin = ((hidden_state >> 16) & 0xFF);

    int bucket = get_merlin_bucket(state, assassin, hidden_state);
    float strategy[5];
    calculate_strategy(merlin_regretsum + 5 * bucket, 5, strategy);

    float correct_pick_prob = strategy[merlin];
    if (traverser != assassin) {
        __sync_fetch_and_add(&games_explored, 1);
        float expected_payoff = (1.0 - correct_pick_prob) - correct_pick_prob;
        expected_payoff = (traverser == (hidden_state & 0xFF)) ? -expected_payoff : expected_payoff;
        return expected_payoff;
    }

    __sync_fetch_and_add(&games_explored, 5);

    // cout << "Making merlin decision" << endl;
    float value = correct_pick_prob + correct_pick_prob - 1.0;

    mtxs[bucket % NUM_MUTEXES].lock();
    for (int i = 0; i < 5; i++) {
        float action_value = (i == merlin) ? 1.0 : -1.0;
        merlin_regretsum[bucket * 5 + i] += (action_value - value) * t;
        merlin_stratsum[bucket * 5 + i] += pi * strategy[i] * t;
    }
    mtxs[bucket % NUM_MUTEXES].unlock();

    return value;
}

float mccfr(int t, const RoundState& state, uint32_t hidden_state, int traverser, float pi) {
    if (state.is_terminal()) {
        __sync_fetch_and_add(&games_explored, 1);
        return state.terminal_payoff(traverser, hidden_state);
    }

    switch (state.status) {
    case Status_propose:
        return mccfr_propose(t, state, hidden_state, traverser, pi);
    case Status_vote:
        return mccfr_vote(t, state, hidden_state, traverser, pi);
    case Status_mission:
        return mccfr_mission(t, state, hidden_state, traverser, pi);
    case Status_merlin:
        return mccfr_merlin(t, state, hidden_state, traverser, pi);
    default:
        // You got a bug.
        assert(false);
    }
}

void allocate_buffers() {
    cout << "Allocating..." << endl;

    proposal_regretsum = (float*) calloc(NUM_PROPOSAL_FLOATS, sizeof(float));
    assert(proposal_regretsum != NULL);
    proposal_stratsum = (float*) calloc(NUM_PROPOSAL_FLOATS, sizeof(float));
    assert(proposal_stratsum != NULL);
    voting_regretsum = (float*) calloc(NUM_VOTING_FLOATS, sizeof(float));
    assert(voting_regretsum != NULL);
    voting_stratsum = (float*) calloc(NUM_VOTING_FLOATS, sizeof(float));
    assert(voting_stratsum != NULL);
    mission_regretsum = (float*) calloc(NUM_MISSION_FLOATS, sizeof(float));
    assert(mission_regretsum != NULL);
    mission_stratsum = (float*) calloc(NUM_MISSION_FLOATS, sizeof(float));
    assert(mission_stratsum != NULL);
    merlin_regretsum = (float*) calloc(NUM_MERLIN_FLOATS, sizeof(float));
    assert(merlin_regretsum != NULL);
    merlin_stratsum = (float*) calloc(NUM_MERLIN_FLOATS, sizeof(float));
    assert(merlin_stratsum != NULL);
}

mutex output_mtx;
void seed_rng() {
    auto thread_id = this_thread::get_id();
    hash<std::thread::id> hasher;
    rngx = (uint32_t) hasher(thread_id);
    int num_refreshes = rngx % 19;
    for (int i = 0; i < num_refreshes; i++) {
        int result = xorshf96();
    }
}

void write_file(const string& filename, void* buffer, size_t buflen) {
    ofstream file;
    file.open(filename);
    file.write((const char*) buffer, buflen);
    file.close();
}


void* read_file(const string& filename) {
    ifstream in(filename, ios::in | ios::binary);
    if (!in) {
        return NULL;
    }
    in.seekg(0, ios::end);
    size_t size = in.tellg();
    void* buffer = malloc(size);
    in.seekg(0, ios::beg);
    in.read((char*) buffer, size);
    in.close();
    return buffer;
}

void save_checkpoint_buffers(int num_iterations, bool save_regrets) {
    cout << "Saving checkpoint files after " << num_iterations << " iterations" << endl;
    write_file(to_string(num_iterations) + "_proposal_stratsum.dat", proposal_stratsum, NUM_PROPOSAL_FLOATS * sizeof(float));
    write_file(to_string(num_iterations) + "_voting_stratsum.dat", voting_stratsum, NUM_VOTING_FLOATS * sizeof(float));
    write_file(to_string(num_iterations) + "_mission_stratsum.dat", mission_stratsum, NUM_MISSION_FLOATS * sizeof(float));
    write_file(to_string(num_iterations) + "_merlin_stratsum.dat", merlin_stratsum, NUM_MERLIN_FLOATS * sizeof(float));
    if (save_regrets) {
        write_file(to_string(num_iterations) + "_proposal_regretsum.dat", proposal_regretsum, NUM_PROPOSAL_FLOATS * sizeof(float));
        write_file(to_string(num_iterations) + "_voting_regretsum.dat", voting_regretsum, NUM_VOTING_FLOATS * sizeof(float));
        write_file(to_string(num_iterations) + "_mission_regretsum.dat", mission_regretsum, NUM_MISSION_FLOATS * sizeof(float));
        write_file(to_string(num_iterations) + "_merlin_regretsum.dat", merlin_regretsum, NUM_MERLIN_FLOATS * sizeof(float));
    }
}

void save_buffers(int num_iterations, bool save_regrets) {
    cout << "Saving files after " << num_iterations << " iterations" << endl;
    write_file("latest_proposal_stratsum.dat", proposal_stratsum, NUM_PROPOSAL_FLOATS * sizeof(float));
    write_file("latest_voting_stratsum.dat", voting_stratsum, NUM_VOTING_FLOATS * sizeof(float));
    write_file("latest_mission_stratsum.dat", mission_stratsum, NUM_MISSION_FLOATS * sizeof(float));
    write_file("latest_merlin_stratsum.dat", merlin_stratsum, NUM_MERLIN_FLOATS * sizeof(float));
    if (save_regrets) {
        write_file(to_string(num_iterations) + "_proposal_regretsum.dat", proposal_regretsum, NUM_PROPOSAL_FLOATS * sizeof(float));
        write_file(to_string(num_iterations) + "_voting_regretsum.dat", voting_regretsum, NUM_VOTING_FLOATS * sizeof(float));
        write_file(to_string(num_iterations) + "_mission_regretsum.dat", mission_regretsum, NUM_MISSION_FLOATS * sizeof(float));
        write_file(to_string(num_iterations) + "_merlin_regretsum.dat", merlin_regretsum, NUM_MERLIN_FLOATS * sizeof(float));
    }
}

void load_checkpoint_buffers(int num_iterations) {
    cout << "Loading checkpoints from iteration " << num_iterations << endl;
    proposal_stratsum = (float*) read_file(to_string(num_iterations) + "_proposal_stratsum.dat");
    voting_stratsum = (float*) read_file(to_string(num_iterations) + "_voting_stratsum.dat");
    mission_stratsum = (float*) read_file(to_string(num_iterations) + "_mission_stratsum.dat");
    merlin_stratsum = (float*) read_file(to_string(num_iterations) + "_merlin_stratsum.dat");
    proposal_regretsum = (float*) read_file(to_string(num_iterations) + "_proposal_regretsum.dat");
    voting_regretsum = (float*) read_file(to_string(num_iterations) + "_voting_regretsum.dat");
    mission_regretsum = (float*) read_file(to_string(num_iterations) + "_mission_regretsum.dat");
    merlin_regretsum = (float*) read_file(to_string(num_iterations) + "_merlin_regretsum.dat");
}

int main() {
//    allocate_buffers();
    load_checkpoint_buffers(4000000);
    int proposal_arr_size = NUM_PROPOSAL_FLOATS * sizeof(float);
    int voting_arr_size = NUM_VOTING_FLOATS * sizeof(float);
    int mission_arr_size = NUM_MISSION_FLOATS * sizeof(float);
    int merlin_arr_size = NUM_MERLIN_FLOATS * sizeof(float);
    cout << "proposal " << 2 * proposal_arr_size / 1024 / 1024 << "MB" << endl;
    cout << "voting " << 2 * voting_arr_size / 1024 / 1024 << "MB" << endl;
    cout << "mission " << 2 * mission_arr_size / 1024 / 1024 << "MB" << endl;
    cout << "merlin " << 2 * merlin_arr_size / 1024 / 1024 << "MB" << endl;

    RoundState state = {};
    for (int t = 4000001;; t++) {
        uint32_t hidden_state = random_hidden();
        cilk_for (int player = 0; player < 5; player++) {
            mccfr(t, state, hidden_state, player, 1.0);
        }
        if (t % 100 == 0) {
            cout << "T=" << t << " Games explored: " << games_explored << endl;
            games_explored = 0;
        }
        if (t % 10000 == 0) {
            save_buffers(t, false);
        }
        if (t % 1000000 == 0) {
            save_checkpoint_buffers(t, true);
        }
        cout.flush();
    }

/*
    while (true) {
        vector<thread> threads(num_cores);
        cout << "Spawning..." << endl;
        for (int i = 0; i < num_cores; i++) {
            threads[i] = std::thread(run_mccfr, t, chunksize);
        }
        cout << "Joining..." << endl;
        for (int i = 0; i < num_cores; i++) {
            threads[i].join();
        }
        t += chunksize;
        total_num_iterations += num_cores * chunksize;
        save_buffers(total_num_iterations, false);
    }*/
}
