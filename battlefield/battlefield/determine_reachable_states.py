import copy
import numpy as np

from battlefield.avalon_types import GOOD_ROLES, EVIL_ROLES, possible_hidden_states, starting_hidden_states, ProposeAction, VoteAction, MissionAction, PickMerlinAction
from battlefield.avalon import AvalonState
from battlefield.helpers import simple_cache

def quantize_belief(belief, granularity=100):
    belief = np.around(belief * 100).astype(int)
    return tuple(belief)


def normalize_belief(quantized):
    result = np.array(quantized)
    return result / np.sum(result)


def determine_reachable_from_belief_state(cache, state, belief, player, hidden_state_to_index):
    if isinstance(belief, np.ndarray):
        belief = quantize_belief(belief)

    if (state, belief, player) in cache:
        return
    cache.add((state, belief, player))

    belief = normalize_belief(belief)


def get_starting_belief(hidden_state, player, hidden_state_to_index):
    belief = np.zeros(len(hidden_state_to_index))

    for hidden_state in starting_hidden_states(player, hidden_state, hidden_state_to_index.keys()):
        belief[hidden_state_to_index[hidden_state]] += 1

    return belief / np.sum(belief)


def determine_reachable(base_bot, roles, num_players):
    hidden_state_to_index = {
        hs: i for i, hs in enumerate(possible_hidden_states(roles, num_players))
    }

    possible = possible_hidden_states(roles, num_players)

    start_state = AvalonState.start_state(num_players)
    cache = {}

    for hidden_state in hidden_state_to_index:
        for player in range(num_players):
            starting_belief = get_starting_belief(hidden_state, player, hidden_state_to_index)
            determine_reachable_from_belief_state(cache, start_state, starting_belief, player, hidden_state_to_index)

    print len(cache)
