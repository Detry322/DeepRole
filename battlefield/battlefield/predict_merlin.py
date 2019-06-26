import pandas as pd
import numpy as np
import copy
import gzip
import itertools
import multiprocessing
import sys


from battlefield.avalon_types import GOOD_ROLES, EVIL_ROLES, possible_hidden_states, starting_hidden_states, ProposeAction, VoteAction, MissionAction, PickMerlinAction
from battlefield.avalon import AvalonState

from battlefield.compare_to_human import reconstruct_hidden_state, load_human_data

def filter_data(human_data, num_players, min_game_id, max_game_id, roles):
    result = []
    for game in human_data:
        hidden_state = reconstruct_hidden_state(game)
        if len(hidden_state) != num_players:
            continue
        if not all(role in roles for role in hidden_state):
            continue
        if game['id'] >= max_game_id or game['id'] < min_game_id:
            continue
        if 'findMerlin' not in game['log'][-1]:
            continue
        result.append(game)

    return result


def handle_transition(state, hidden_state, moves, assassin_bot, assassin_player):
    new_state, _, observation = state.transition(moves, hidden_state)

    my_move = None
    if assassin_player in state.moving_players():
        my_move = moves[state.moving_players().index(assassin_player)]

    assassin_bot.handle_transition(state, new_state, observation, move=my_move)
    return new_state


def handle_round(game, state, hidden_state, assassin_bot, assassin_player, round_):
    last_proposal = None
    for proposal_num in ['1', '2', '3', '4', '5']:
        proposal = last_proposal = round_[proposal_num]
        assert state.proposer == proposal['proposer']
        assert state.propose_count == int(proposal_num) - 1
        moves = [ProposeAction(proposal=tuple(sorted(proposal['team'])))]
        state = handle_transition(state, hidden_state, moves, assassin_bot, assassin_player)

        assert state.status == 'vote'
        moves = [VoteAction(up=(vote == 'Approve')) for vote in proposal['votes']]
        state = handle_transition(state, hidden_state, moves, assassin_bot, assassin_player)

        if state.status == 'run':
            break

    secret_votes = sorted(zip(last_proposal['team'], round_['mission']))
    moves = [MissionAction(fail=(vote == "Fail")) for player, vote in secret_votes]
    state = handle_transition(state, hidden_state, moves, assassin_bot, assassin_player)
    return state


def get_bot_merlin_prediction(bot_class, game):
    hidden_state = reconstruct_hidden_state(game)
    state = AvalonState.start_state(len(hidden_state))

    possible = possible_hidden_states(set(hidden_state), num_players=len(hidden_state))
    perspectives = [
        starting_hidden_states(player, hidden_state, possible)
        for player, _ in enumerate(hidden_state)
    ]

    assassin_player = hidden_state.index('assassin')
    assassin_perspective = perspectives[assassin_player]
    assassin_bot = bot_class.create_and_reset(state, assassin_player, 'assassin', assassin_perspective)

    for round_ in game['log']:
        state = handle_round(game, state, hidden_state, assassin_bot, assassin_player, round_)

    final_round = game['log'][-1]

    assert 'findMerlin' in final_round
    find_merlin = round_['findMerlin']
    assert find_merlin['assassin'] == assassin_player

    legal_moves = state.legal_actions(assassin_player, hidden_state)
    move_probs = assassin_bot.get_move_probabilities(state, legal_moves)

    return {
        'human_guess': find_merlin['merlin_guess'],
        'bot_human_prob': move_probs[find_merlin['merlin_guess']],
        'correct_guess': hidden_state.index('merlin'),
        'bot_correct_prob': move_probs[hidden_state.index('merlin')],
        'top_pick': np.argmax(move_probs),
        'top_pick_prob': np.max(move_probs),
        'game': game['id'],
        'merlin': game['players'][hidden_state.index('merlin')]['player_id'],
        'assassin': game['players'][hidden_state.index('assassin')]['player_id']
    }


BASIC_ROLES = set(['merlin', 'assassin', 'minion', 'servant'])    


def get_merlin_prediction(bot_class, num_players=5, min_game_id=0, max_game_id=50000, roles=BASIC_ROLES):
    data = load_human_data()
    data = filter_data(data, num_players, min_game_id, max_game_id, roles)

    result = []

    for game in data:
        try:
            result.append(get_bot_merlin_prediction(bot_class, game))
        except AssertionError:
            pass

    result = pd.DataFrame(result)
    return result
    
