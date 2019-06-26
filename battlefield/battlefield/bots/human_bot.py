import random
import os
import json
import gzip
import cPickle as pickle
import numpy as np
from collections import defaultdict

from battlefield.bots.bot import Bot
from battlefield.bots.observe_beater_bot import get_python_perspective, get_hidden_states_bucket
from battlefield.compare_to_human import reconstruct_hidden_state, load_human_data as load_human_json
from battlefield.avalon_types import filter_hidden_states, EVIL_ROLES, GOOD_ROLES, VoteAction, ProposeAction, MissionAction, PickMerlinAction
from battlefield.avalon import AvalonState

GAME_DATAFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'data.pkl.gz'))
STRATEGY_DATAFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'human_strat.pkl'))

tree_root = None

def perspective_from_hidden_states(hidden_states):
    roles = [set([]) for _ in range(len(hidden_states[0]))]
    for hidden_state in hidden_states:
        for p, role in enumerate(hidden_state):
            roles[p].add(role)
    return tuple([ frozenset(possible) for possible in roles ])


def load_human_data():
    global tree_root
    if tree_root is not None:
        return tree_root

    print "Loading human data"
    with gzip.open(GAME_DATAFILE, 'r') as f:
        tree_root = pickle.load(f)

    return tree_root


class HumanBot(Bot):
    def __init__(self):
        pass


    def reset(self, game, player, role, hidden_states):
        self.game = game
        self.player = player
        self.role = role
        self.hidden_states = hidden_states
        self.is_evil = role in EVIL_ROLES
        self.data = load_human_data()
        self.current_node = self.data.get((player, perspective_from_hidden_states(hidden_states)))


    def handle_transition(self, old_state, new_state, observation, move=None):
        if self.current_node is not None:
            self.current_node = self.current_node['transitions'].get((new_state.as_key(), observation, move))
            if self.current_node is None:
                print "Warning: exited human play...", old_state, new_state, observation, move


    def get_action(self, state, legal_actions):
        if self.current_node is None:
            if state.status == 'run':
                return MissionAction(fail=self.is_evil)
            if state.status == 'vote':
                return VoteAction(up=True)
            return random.choice(legal_actions)

        return max(self.current_node['move_counts'], key=self.current_node['move_counts'].get)


    def get_move_probabilities(self, state, legal_actions):
        raise NotImplemented



def game_state_generator(avalon_start, human_game, hidden_state):
    # at each step, return old state, new state, and observation
    state = avalon_start

    for round_ in human_game['log']:
        last_proposal = None
        for proposal_num in ['1', '2', '3', '4', '5']:
            proposal = last_proposal = round_[proposal_num]
            assert state.proposer == proposal['proposer']
            assert state.propose_count == int(proposal_num) - 1
            moves = [ProposeAction(proposal=tuple(sorted(proposal['team'])))]
            new_state, _, observation = state.transition(moves, hidden_state)
            yield state, moves
            state = new_state

            assert state.status == 'vote'
            moves = [VoteAction(up=(vote == 'Approve')) for vote in proposal['votes']]
            new_state, _, observation = state.transition(moves, hidden_state)
            yield state, moves
            state = new_state

            if state.status == 'run':
                break

        secret_votes = sorted(zip(last_proposal['team'], round_['mission']))
        moves = [MissionAction(fail=(vote == "Fail")) for player, vote in secret_votes]
        new_state, _, observation = state.transition(moves, hidden_state)
        yield state, moves
        state = new_state

        if state.status == 'merlin':
            assert 'findMerlin' in round_
            yield state, [ PickMerlinAction(merlin=round_['findMerlin']['merlin_guess']) ] * 5


def zeros_5():
    return np.zeros(5)


def zeros_10():
    return np.zeros(10)


def zeros_2():
    return np.zeros(2)


counts = None
def read_data_load_counts():
    global counts
    if counts is not None:
        return counts

    if counts is None and os.path.isfile(STRATEGY_DATAFILE):
        with open(STRATEGY_DATAFILE, 'r') as f:
            counts = pickle.load(f)
        return counts

    print "Creating counts data..."
    counts = {
        'propose': defaultdict(zeros_10),
        'vote': defaultdict(zeros_2),
        'run': defaultdict(zeros_2),
        'merlin': defaultdict(zeros_5)
    }

    human_games = load_human_json()
    for game in human_games:
        hidden_state = reconstruct_hidden_state(game)
        if len(hidden_state) != 5 or frozenset(hidden_state) != frozenset(['merlin', 'assassin', 'minion', 'servant']):
            continue
        print game['id']
        start_state = AvalonState.start_state(len(hidden_state))
        fails = []
        perspectives = [
            get_python_perspective(hidden_state, player) for player in range(len(hidden_state))
        ]
        for state, moves in game_state_generator(start_state, game, hidden_state):
            moving_players = state.moving_players()
            for player, move in zip(moving_players, moves):
                legal_actions = state.legal_actions(player, hidden_state)
                perspective_bucket = get_hidden_states_bucket(perspectives[player], fails)
                index = legal_actions.index(move)
                counts[state.status][(state.as_key(), perspectives[player], perspective_bucket)][index] += 1

            if state.status == 'run' and any([move.fail for move in moves ]):
                fails.append((state.proposal, len([move for move in moves if move.fail ])))


    print "Writing counts data..."
    with open(STRATEGY_DATAFILE, 'w') as f:
        pickle.dump(counts, f)

    return counts


class HumanLikeBot(Bot):
    def __init__(self, counts=None):
        self.counts = read_data_load_counts() if counts is None else counts
        self.player = None
        self.fails = None
        self.my_perspective = None

    def __deepcopy__(self, memo):
        new = HumanLikeBot(counts=self.counts)
        new.player = self.player
        new.fails = self.fails[:]
        new.my_perspective = self.my_perspective
        return new

    def reset(self, game, player, role, hidden_states):
        self.player = player
        self.fails = []
        self.my_perspective = get_python_perspective(hidden_states[0], player)


    def handle_transition(self, old_state, new_state, observation, move=None):
        if old_state == 'run' and observation > 0:
            self.fails.append((old_state.proposal, observation))


    def get_action(self, state, legal_actions):
        probs = self.get_move_probabilities(state, legal_actions)
        index = np.random.choice(len(legal_actions), p=probs)
        return legal_actions[index]


    def get_move_probabilities(self, state, legal_actions):
        if len(legal_actions) == 1:
            return np.ones(1)

        perspective_bucket = get_hidden_states_bucket(self.my_perspective, self.fails)
        counts = self.counts[state.status][(state.as_key(), self.my_perspective, perspective_bucket)]

        return (counts + 1.0) / np.sum(counts + 1.0)

