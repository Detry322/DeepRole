import random
import os
import json
import gzip
import cPickle as pickle
import numpy as np

from battlefield.bots.bot import Bot
from battlefield.avalon_types import filter_hidden_states, EVIL_ROLES, GOOD_ROLES, VoteAction, ProposeAction, MissionAction

DATAFILE_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'cfr'))

LOADED_DATAFILES = {}

def load_datafile(filename):
    global LOADED_DATAFILES
    if filename in LOADED_DATAFILES:
        return LOADED_DATAFILES[filename]

    with open(filename) as f:
        LOADED_DATAFILES[filename] = np.fromfile(f, dtype=np.float32)

    return LOADED_DATAFILES[filename]


def load_datafiles(iteration):
    return [
        load_datafile(os.path.join(DATAFILE_BASE, '{}_proposal_stratsum.dat'.format(iteration))),
        load_datafile(os.path.join(DATAFILE_BASE, '{}_voting_stratsum.dat'.format(iteration))),
        load_datafile(os.path.join(DATAFILE_BASE, '{}_mission_stratsum.dat'.format(iteration))),
        load_datafile(os.path.join(DATAFILE_BASE, '{}_merlin_stratsum.dat'.format(iteration))),
    ]

HISTORY_SIZE = 162 * 21 * 5
MERLIN_HISTORY_SIZE = 162 * 243
NUM_EVIL_VIEWPOINTS = 20
NUM_POSSIBLE_VIEWPOINTS = NUM_EVIL_VIEWPOINTS + 30 + 5
PROPOSAL_FAIL_LOOKUP = [0, 1, 2, 3, 4, -1, 5, -1, -1, 6, 7, -1, 8, -1, -1, -1, -1, -1, 9, -1, -1, -1, -1, -1, -1, -1, -1, 10, 11, -1, 12, -1, -1, -1, -1, -1, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 16, -1, 17, -1, -1, -1, -1, -1, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 19, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 20, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
MISSION_FAIL_LOOKUP = [0, -1, -1, -1, 1, -1, -1, -1, 2, -1, 3, -1, 4, 5, 6, -1, 7, 8, -1, -1, 9, -1, 10, 11, 12, 13, 14, -1, 15, -1, 16, 17, 18, -1, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, -1, 30, 31, 32, 33, 34, 35, 36, -1, -1, -1, 37, -1, 38, 39, 40, 41, 42, -1, 43, 44, 45, 46, 47, 48, 49, -1, 50, 51, 52, 53, 54, -1, 55, -1, -1, -1, 56, -1, 57, 58, 59, -1, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, -1, 71, 72, 73, 74, 75, 76, 77, -1, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, -1, 95, 96, 97, 98, 99, -1, 100, -1, -1, -1, 101, 102, 103, 104, 105, 106, 107, -1, 108, 109, 110, 111, 112, -1, 113, -1, -1, 114, 115, -1, 116, -1, -1, -1, -1, -1, -1, -1, 117, -1, 118, 119, 120, 121, 122, -1, 123, 124, 125, 126, 127, 128, 129, -1, 130, 131, 132, 133, 134, -1, 135, -1, -1, -1, 136, 137, 138, 139, 140, 141, 142, -1, 143, 144, 145, 146, 147, -1, 148, -1, -1, 149, 150, -1, 151, -1, -1, -1, -1, -1, 152, 153, 154, 155, 156, -1, 157, -1, -1, 158, 159, -1, 160, -1, -1, -1, -1, -1, 161, -1, -1, -1, -1, -1, -1, -1, -1]
EVIL_LOOKUP = [-1, 0, 2, 4, 6, 1, -1, 8, 10, 12, 3, 9, -1, 14, 16, 5, 11, 15, -1, 18, 7, 13, 17, 19, -1]
MERLIN_LOOKUP = [-1, -1, -1, -1, 0, 1, 2, 3, 4, 5, -1, 6, 7, 8, -1, -1, -1, 9, 10, 11, 12, -1, 13, 14, -1, 15, 16, -1, -1, 17, 18, 19, -1, 20, 21, -1, 22, -1, 23, -1, 24, 25, 26, -1, 27, 28, -1, 29, -1, -1]
PROPOSAL_TO_INDEX_LOOKUP = [-1, -1, -1, 0, -1, 1, 4, 0, -1, 2, 5, 1, 7, 3, 6, -1, -1, 3, 6, 2, 8, 4, 7, -1, 9, 5, 8, -1, 9, -1, -1, -1]
INDEX_TO_PROPOSAL_2 = [3, 5, 9, 17, 6, 10, 18, 12, 20, 24]
INDEX_TO_PROPOSAL_3 = [7, 11, 19, 13, 21, 25, 14, 22, 26, 28]
ROUND_TO_PROPOSE_SIZE = [2, 3, 2, 3, 3]


def proposal_to_bitstring(proposal):
    result = 0
    for p in proposal:
        result |= (1 << p)
    assert result < 32
    return result


def bitstring_to_proposal(bitstring):
    result = ()
    i = 0
    for i in range(5):
        if ((1 << i) & bitstring) != 0:
            result = result + (i, )
    assert len(result) in [2, 3]
    return result


def get_history_bucket(bot, state):
    proposals_with_fail_big_index = (
        1  * bot.propose_fail_count[0] +
        3  * bot.propose_fail_count[1] +
        9  * bot.propose_fail_count[2] +
        27 * bot.propose_fail_count[3] +
        81 * bot.propose_fail_count[4]
    )
    missions_with_fail_big_index = (
        1  * bot.mission_fail_count[0] +
        3  * bot.mission_fail_count[1] +
        9  * bot.mission_fail_count[2] +
        27 * bot.mission_fail_count[3] +
        81 * bot.mission_fail_count[4]
    )
    proposals_with_fail_index = PROPOSAL_FAIL_LOOKUP[proposals_with_fail_big_index]
    assert proposals_with_fail_index >= 0
    missions_with_fail_index = MISSION_FAIL_LOOKUP[missions_with_fail_big_index]
    assert missions_with_fail_index >= 0
    return (162 * 5 * proposals_with_fail_index) + (5 * missions_with_fail_index) + (state.fails + state.succeeds)


def get_merlin_history_bucket(bot, state):
    missions_with_fail_big_index = (
        1  * bot.mission_fail_count[0] +
        3  * bot.mission_fail_count[1] +
        9  * bot.mission_fail_count[2] +
        27 * bot.mission_fail_count[3] +
        81 * bot.mission_fail_count[4]
    )
    missions_with_fail_index = MISSION_FAIL_LOOKUP[missions_with_fail_big_index]
    assert missions_with_fail_index >= 0
    upvoted_fail = (
        1  * bot.vote_up_fail_count[0] +
        3  * bot.vote_up_fail_count[1] +
        9  * bot.vote_up_fail_count[2] +
        27 * bot.vote_up_fail_count[3] +
        81 * bot.vote_up_fail_count[4]
    )
    return (243 * missions_with_fail_index) + upvoted_fail


num_decision_points = 0
num_random_decisions = 0


class _CFRBot(Bot):
    ITERATION = None

    def __init__(self):
        pass

    def reset(self, game, player, role, hidden_states):
        assert len(hidden_states[0]) == 5, "CFRBot can only play 5 player"
        assert self.ITERATION is not None, "Can't load datafiles for none"
        self.vote_up_fail_count = np.array([0]*5)
        self.propose_fail_count = np.array([0]*5)
        self.mission_fail_count = np.array([0]*5)
        self.old_votes = None
        self.player = player
        self.role = role
        if role == 'merlin' or role in EVIL_ROLES:
            self.bad = tuple(sorted([ i for i, role in enumerate(hidden_states[0]) if role in EVIL_ROLES ]))
            assert len(self.bad) == 2
        else:
            self.bad = None 

    @property
    def PROPOSAL_STRAT(self):
        return load_datafile(os.path.join(DATAFILE_BASE, '{}_proposal_stratsum.dat'.format(self.ITERATION)))

    @property
    def VOTING_STRAT(self):
        return load_datafile(os.path.join(DATAFILE_BASE, '{}_voting_stratsum.dat'.format(self.ITERATION)))

    @property
    def MISSION_STRAT(self):
        return load_datafile(os.path.join(DATAFILE_BASE, '{}_mission_stratsum.dat'.format(self.ITERATION)))

    @property
    def MERLIN_STRAT(self):
        return load_datafile(os.path.join(DATAFILE_BASE, '{}_merlin_stratsum.dat'.format(self.ITERATION)))


    def get_general_perspective(self):
        if self.role == 'merlin':
            bad_guy_index = EVIL_LOOKUP[5 * self.bad[0] + self.bad[1]]
            assert bad_guy_index >= 0
            merlin_perspective = MERLIN_LOOKUP[10 * self.player + (bad_guy_index / 2)]
            assert merlin_perspective >= 0
            return merlin_perspective + 25
        elif self.role in EVIL_ROLES:
            partner = self.bad[1] if self.player == self.bad[0] else self.bad[0]
            perspective = EVIL_LOOKUP[5 * self.player + partner]
            assert perspective >= 0
            return perspective + 5
        else:
            return self.player


    def get_propose_bucket(self, state):
        perspective = self.get_general_perspective()
        propose_num = state.propose_count / 2
        history = get_history_bucket(self, state)
        index = (NUM_POSSIBLE_VIEWPOINTS * 3 * history) + (3 * perspective) + propose_num
        return index


    def get_vote_bucket(self, state):
        perspective = self.get_general_perspective()
        propose_num = state.propose_count / 2
        history = get_history_bucket(self, state)
        proposal = PROPOSAL_TO_INDEX_LOOKUP[proposal_to_bitstring(state.proposal)]
        assert proposal >= 0
        index = (
            (NUM_POSSIBLE_VIEWPOINTS * 3 * 10 * history) +
            (3 * 10 * perspective) +
            (10 * propose_num) +
            proposal
        )
        return index


    def get_mission_bucket(self, state):
        assert self.role in EVIL_ROLES
        partner = self.bad[1] if self.player == self.bad[0] else self.bad[0]
        perspective = EVIL_LOOKUP[5 * self.player + partner]
        history = get_history_bucket(self, state)
        proposal = PROPOSAL_TO_INDEX_LOOKUP[proposal_to_bitstring(state.proposal)]
        assert proposal >= 0
        index = (
            (10 * HISTORY_SIZE * perspective) +
            (10 * history) +
            proposal
        )
        return index


    def get_merlin_bucket(self, state):
        assert self.role == 'assassin'
        perspective = EVIL_LOOKUP[5 * self.bad[0] + self.bad[1]] / 2
        history = get_merlin_history_bucket(self, state)
        index = (
            (MERLIN_HISTORY_SIZE * perspective) +
            history
        )
        return index


    def handle_transition(self, old_state, new_state, observation, move=None):
        if old_state.status == 'vote':
            self.old_votes = np.array([1 if ob.up else 0 for ob in observation])
        if old_state.fails < new_state.fails:
            self.vote_up_fail_count += self.old_votes
            self.propose_fail_count[old_state.proposer] += 1
            self.mission_fail_count[list(old_state.proposal)] += 1


    def get_action(self, state, legal_actions):
        probs = self.get_move_probabilities(state, legal_actions)
        index = np.random.choice(len(legal_actions), p=probs)
        return legal_actions[index]


    def get_move_probabilities(self, state, legal_actions):
        if len(legal_actions) == 1:
            return np.array([1.0])

        if state.status == 'merlin' and self.role != 'assassin':
            return np.array([1.0/len(legal_actions)] * len(legal_actions))

        if state.status == 'propose':
            bucket = self.get_propose_bucket(state)
            strat = self.PROPOSAL_STRAT[10*bucket:10*(bucket + 1)]
            propose_size = ROUND_TO_PROPOSE_SIZE[state.fails + state.succeeds]
            PROPOSAL_BITSTRING_LOOKUP = INDEX_TO_PROPOSAL_2 if propose_size == 2 else INDEX_TO_PROPOSAL_3
            strat_prob = {}
            for i, p in enumerate(strat):
                strat_prob[bitstring_to_proposal(PROPOSAL_BITSTRING_LOOKUP[i])] = p
            assert len(strat_prob) == 10
            unnormalized = np.array([ strat_prob[move.proposal] for move in legal_actions ])
        elif state.status == 'vote':
            bucket = self.get_vote_bucket(state)
            strat = self.VOTING_STRAT[2*bucket:2*(bucket + 1)]
            unnormalized = np.array([ strat[int(vote.up)] for vote in legal_actions ])
        elif state.status == 'run':
            bucket = self.get_mission_bucket(state)
            strat = self.MISSION_STRAT[2*bucket:2*(bucket + 1)]
            unnormalized = np.array([ strat[int(act.fail)] for act in legal_actions ])
        elif state.status == 'merlin':
            bucket = self.get_merlin_bucket(state)
            unnormalized = self.MERLIN_STRAT[5*bucket:5*(bucket + 1)]
        else:
            assert False


        if np.sum(unnormalized) == 0.0:
            return np.array([1.0/len(legal_actions)] * len(legal_actions))

        return unnormalized / np.sum(unnormalized) 


class _CFRBotCreator:
    def __init__(self, iteration):
        self.__name__ = "CFRBot_{}".format(iteration)
        self.iteration = iteration
        self.cls = None

    def __call__(self, *args, **kwargs):
        if self.cls is not None:
            return self.cls(*args, **kwargs)
        name = "CFRBot_{}".format(self.iteration)
        class IterCFRBot(_CFRBot):
            __name__ = name
            ITERATION = str(self.iteration)
        IterCFRBot.__name__ = name
        self.cls = IterCFRBot
        return self.cls(*args, **kwargs)

    def create_and_reset(self, game, player, role, hidden_states):
        bot = self()
        bot.reset(game, player, role, hidden_states)
        return bot


def CFRBot(iteration='latest'):
    return _CFRBotCreator(iteration)
