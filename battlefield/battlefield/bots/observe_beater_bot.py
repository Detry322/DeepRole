import random
import numpy as np
import itertools
import os
import cPickle as pickle
from collections import defaultdict

from battlefield.bots.bot import Bot
from battlefield.bots.observe_bot import ObserveBot
from battlefield.avalon_types import starting_hidden_states, possible_hidden_states, EVIL_ROLES, GOOD_ROLES, ProposeAction, MissionAction, VoteAction, PickMerlinAction, filter_hidden_states
from battlefield.bots.cfr_bot import EVIL_LOOKUP, MERLIN_LOOKUP, PROPOSAL_TO_INDEX_LOOKUP, proposal_to_bitstring, bitstring_to_proposal, INDEX_TO_PROPOSAL_2, INDEX_TO_PROPOSAL_3


def get_python_perspective(hidden_state, player):
    evil = tuple(sorted([i for i in range(len(hidden_state)) if hidden_state[i] in EVIL_ROLES]))
    if hidden_state[player] == 'merlin':
        return ('merlin', player, evil)
    elif hidden_state[player] in EVIL_ROLES:
        return ('evil', player, evil)
    else:
        return ('player', player, None)


perspective_fails_to_bucket = {}
bucket_to_hidden_states = []
def get_hidden_states_bucket(player_perspective, fails):
    global perspective_fails_to_bucket
    global bucket_to_hidden_states
    key = (player_perspective, tuple(fails))
    if key in perspective_fails_to_bucket:
        return perspective_fails_to_bucket[key]

    print "Calculating big lookup table..."
    all_hidden_states = possible_hidden_states(['merlin', 'minion', 'servant', 'assassin'], 5)
    possible_beliefs_to_key = defaultdict(lambda: [])

    for roles in all_hidden_states:
        beliefs = [
            starting_hidden_states(player, roles, all_hidden_states) for player in range(5)
        ]
        perspectives = [
            get_python_perspective(roles, player) for player in range(5)
        ]

        for perspective, b in zip(perspectives, beliefs):
            possible_beliefs_to_key[frozenset(b)].append((perspective, ()))

        for p1size in [2,3]:
            for p1 in itertools.combinations(range(5), p1size):
                num_bad = len([ p for p in p1 if roles[p] in EVIL_ROLES ])
                if num_bad == 0:
                    continue
                for num_fail in range(1, num_bad+1):
                    new_beliefs = [ filter_hidden_states(b, p1, num_fail) for b in beliefs ]
                    for perspective, b in zip(perspectives, new_beliefs):
                        possible_beliefs_to_key[frozenset(b)].append(
                            (perspective, ( (p1, num_fail), ))
                        )

                    for p2size in [2,3]:
                        for p2 in itertools.combinations(range(5), p2size):
                            num_bad2 = len([ p for p in p2 if roles[p] in EVIL_ROLES ])
                            if num_bad2 == 0:
                                continue
                            for num_fail2 in range(1, num_bad2+1):
                                for perspective, b in zip(perspectives, new_beliefs):
                                    b_prime = filter_hidden_states(b, p2, num_fail2)
                                    possible_beliefs_to_key[frozenset(b_prime)].append(
                                        (perspective, ((p1, num_fail), (p2, num_fail2)))
                                    )
    for i, (hidden_states, keys) in enumerate(possible_beliefs_to_key.iteritems()):
        assert len(bucket_to_hidden_states) == i
        bucket_to_hidden_states.append(list(hidden_states))
        for key in keys:
            perspective_fails_to_bucket[key] = i

    return get_hidden_states_bucket(player_perspective, fails)



observebot_move_probs = {}
def calculate_observebot_move_probs(state, perspective, bucket):
    global observebot_move_probs
    global bucket_to_hidden_states
    key = (state.as_key(), perspective, bucket)

    if key not in observebot_move_probs:
        hidden_states = bucket_to_hidden_states[bucket]
        role = hidden_states[0][perspective[1]]
        ob = ObserveBot()
        ob.reset(state, perspective[1], role, hidden_states)

        legal_actions = state.legal_actions(perspective[1], hidden_states[0])
        observebot_move_probs[key] = ob.get_move_probabilities(state, legal_actions)

    return observebot_move_probs[key]


def zeros_5():
    return np.zeros(5)

def zeros_10():
    return np.zeros(10)

def zeros_2():
    return np.zeros(2)

class ObserveBeaterBot(Bot):
    def __init__(self):
        self.game_num = 0

        self.cfr_regret = {
            'merlin': defaultdict(zeros_5),
            'propose': defaultdict(zeros_10),
            'vote': defaultdict(zeros_2),
            'run': defaultdict(zeros_2),
        }
        self.cfr_strat = {
            'merlin': defaultdict(zeros_5),
            'propose': defaultdict(zeros_10),
            'vote': defaultdict(zeros_2),
            'run': defaultdict(zeros_2),
        }


    def reset(self, game, player, role, hidden_states):
        if self.game_num == 0:
            with open('1000000_0.15_strat.pkl') as f:
                self.cfr_strat = pickle.load(f)
                
            # num_iterations = 1000000
            # print "Training for {} iterations...".format(num_iterations)

            # hidden_state = ['minion', 'merlin', 'servant', 'servant', 'assassin']
            # for i in range(num_iterations):
            #     print i
            #     search_player = np.random.choice(5)
            #     random.shuffle(hidden_state)
            #     # hidden_state = random.choice(hidden_states)
            #     self.cfr_search(search_player, game, tuple(hidden_state), [], 1.0, i + 1)

            # with open('1000000_0.15_regret.pkl', 'w') as f:
            #     pickle.dump(self.cfr_regret, f)

            # with open('1000000_0.15_strat.pkl', 'w') as f:
            #     pickle.dump(self.cfr_strat, f)


        self.player = player
        self.hidden_states = hidden_states
        self.fails = []
        self.role = role
        self.game_num += 1
        self.my_perspective = get_python_perspective(hidden_states[0], self.player)


    def handle_transition(self, old_state, new_state, observation, move=None):
        if old_state.status == 'run' and observation > 0:
            self.fails.append((old_state.proposal, observation))


    def set_bot_ids(self, bot_ids):
        self.bot_ids = bot_ids


    def cfr_search(self, me, state, hidden_state, fails, strategy_probability, t):
        if state.is_terminal():
            return state.terminal_value(hidden_state)[me]

        moving_players = state.moving_players()
        my_move_index = None
        moves = [None] * len(moving_players)
        for i in range(len(moving_players)):
            player = moving_players[i]
            if hidden_state[player] not in EVIL_ROLES and state.status == 'run':
                moves[i] = MissionAction(fail=False)
                continue

            if hidden_state[player] != 'assassin' and state.status == 'merlin':
                moves[i] = PickMerlinAction(merlin=np.random.choice(len(hidden_state)))
                continue

            perspective = get_python_perspective(hidden_state, player)
            perspective_bucket = get_hidden_states_bucket(perspective, fails)

            if player == me:
                if np.random.random() < 0.15:
                    my_move_index = i
                    continue
                strat = self.cfr_regret[state.status][(state.as_key(), perspective, perspective_bucket)]
                strat = np.clip(strat, 0.0, None)
                if np.sum(strat) == 0:
                    strat = np.ones(len(strat))
                move_probs = strat / np.sum(strat)
            else:
                move_probs = calculate_observebot_move_probs(state, perspective, perspective_bucket)

            legal_actions = state.legal_actions(player, hidden_state)
            moves[i] = legal_actions[np.random.choice(len(legal_actions), p=move_probs)]


        if my_move_index is None:
            value = 0.0
            new_state, _, observation = state.transition(moves, hidden_state)
            if state.status == 'run' and observation > 0:
                fails.append((state.proposal, observation))
            value = self.cfr_search(me, new_state, hidden_state, fails, strategy_probability, t)
            if state.status == 'run' and observation > 0:
                fails.pop()
            return value


        perspective = get_python_perspective(hidden_state, me)
        perspective_bucket = get_hidden_states_bucket(perspective, fails)
        my_strategy = self.cfr_regret[state.status][(state.as_key(), perspective, perspective_bucket)]
        my_strategy = np.clip(my_strategy, 0, None)

        if np.sum(my_strategy) == 0:
            p = np.ones(len(my_strategy))/len(my_strategy)
        else:
            p = my_strategy / np.sum(my_strategy)

        values = np.zeros(len(my_strategy))

        legal_actions = state.legal_actions(me, hidden_state)
        for action_index in range(len(values)):
            moves[my_move_index] = legal_actions[action_index]
            new_state, _, observation = state.transition(moves, hidden_state)
            if state.status == 'run' and observation > 0:
                fails.append((state.proposal, observation))
            values[action_index] = self.cfr_search(me, new_state, hidden_state, fails, strategy_probability * p[action_index], t)
            if state.status == 'run' and observation > 0:
                fails.pop()

        strategy_value = np.dot(values, p)
        regrets = values - strategy_value
        key = (state.as_key(), perspective, perspective_bucket)
        self.cfr_regret[state.status][key] += regrets * t
        self.cfr_strat[state.status][key] += p * strategy_probability * t
        return strategy_value


    def get_action(self, state, legal_actions):
        probs = self.get_move_probabilities(state, legal_actions)
        index = np.random.choice(len(legal_actions), p=probs)
        return legal_actions[index]


    def get_move_probabilities(self, state, legal_actions):
        if len(legal_actions) == 1:
            return np.array([1.0])

        perspective_bucket = get_hidden_states_bucket(self.my_perspective, self.fails)

        stratsum = self.cfr_strat[state.status][(state.as_key(), self.my_perspective, perspective_bucket)]
        if np.sum(stratsum) == 0.0:
            return np.ones(len(legal_actions)) / len(legal_actions)

        return stratsum / np.sum(stratsum)
