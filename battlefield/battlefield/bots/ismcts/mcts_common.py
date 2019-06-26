import numpy as np
import random
from battlefield.avalon_types import GOOD_ROLES

def determinization_iterator(possible_hidden_states, num_iterations):
    i = 0
    hidden = list(possible_hidden_states)
    while i < num_iterations:
        random.shuffle(hidden)
        for h in hidden:
            if i >= num_iterations:
                return
            yield i, h
            i += 1



def random_choice(values, p=None):
    return values[np.random.choice(range(len(values)), p=p)]


def simulate(game_state, hidden_state):
    while not game_state.is_terminal():
        moves = tuple([
            random_choice(game_state.legal_actions(player, hidden_state))
            for player in game_state.moving_players()
        ])
        game_state, hidden_state, _ = game_state.transition(moves, hidden_state)
    return game_state.terminal_value(hidden_state)


# Assumes each mission has a 50-50 chance of failing or succeeding
GOOD_PASS_THREE_PROB = {
    (0, 0): 0.5,
    (0, 1): 0.3125,
    (0, 2): 0.125,
    (1, 0): 0.6875,
    (1, 1): 0.5,
    (1, 2): 0.25,
    (2, 0): 0.875,
    (2, 1): 0.75,
    (2, 2): 0.5
}

def heuristic_value_func(state, hidden_state, player):
    good_win_payoff = 1.0 if hidden_state[player] in GOOD_ROLES else -float(state.NUM_GOOD)/state.NUM_EVIL
    if state.succeeds == 3:
        good_pass_three_prob = 1.0
    elif state.fails == 3:
        good_pass_three_prob = 0.0
    else:
        good_pass_three_prob = GOOD_PASS_THREE_PROB[(state.succeeds, state.fails)]

    find_merlin_prob = 1.0/state.NUM_GOOD
    good_win_prob = (1 - find_merlin_prob) * good_pass_three_prob + find_merlin_prob * 0.0
    return good_win_payoff*good_win_prob + (-good_win_payoff)*(1 - good_win_prob)


def fast_simulate(state, hidden_state):
    return np.array([ heuristic_value_func(state, hidden_state, p) for p in range(len(hidden_state))])
