import os
import gzip
import cPickle as pickle
import json
import sys
import pandas as pd

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.append(PARENT_DIR)

from battlefield.avalon import create_avalon_game
from battlefield.avalon_types import EVIL_ROLES, GOOD_ROLES, ProposeAction, VoteAction, MissionAction, PickMerlinAction, possible_hidden_states, starting_hidden_states

DATAFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'relabeled.json'))

def reconstruct_hidden_state(game):
    roles = []
    for player in game['players']:
        roles.append('minion' if player['spy'] else 'servant')

    for role, p in game['roles'].items():
        if role not in set(['assassin', 'merlin', 'mordred', 'percival', 'morgana', 'oberon']):
            continue
        roles[p] = role
    return tuple(roles)


def handle_round(data, state, hidden_state, game, round_):
    last_proposal = None
    for proposal_num in ['1', '2', '3', '4', '5']:
        proposal = last_proposal = round_[proposal_num]
        assert state.proposer == proposal['proposer'], "idk"
        assert state.propose_count == int(proposal_num) - 1, "idk2"
        data.append({
            'game': game['id'],
            'player': game['players'][proposal['proposer']]['player_id'],
            'seat': proposal['proposer'],
            'role': hidden_state[proposal['proposer']],
            'is_evil': hidden_state[proposal['proposer']] in EVIL_ROLES,
            'type': 'propose',
            'move': ','.join(map(str, sorted(proposal['team']))),
            'propose_count': state.propose_count,
            'round': state.fails + state.succeeds,
            'succeeds': state.succeeds,
            'fails': state.fails,
            'propose_has_self': proposal['proposer'] in proposal['team'],
            'num_players': len(hidden_state)
        })

        moves = [ProposeAction(proposal=tuple(sorted(proposal['team'])))]
        state, _, _ = state.transition(moves, hidden_state)
        assert state.status == 'vote'
        for seat, vote in enumerate(proposal['votes']):
            data.append({
                'game': game['id'],
                'player': game['players'][seat]['player_id'],
                'seat': seat,
                'role': hidden_state[seat],
                'is_evil': hidden_state[seat] in EVIL_ROLES,
                'type': 'vote',
                'move': vote,
                'propose_count': state.propose_count,
                'round': state.fails + state.succeeds,
                'succeeds': state.succeeds,
                'fails': state.fails,
                'propose_has_self': seat in proposal['team'],
                'num_players': len(hidden_state)
            })
        moves = [VoteAction(up=(vote == 'Approve')) for vote in proposal['votes']]
        state, _, _ = state.transition(moves, hidden_state)
        if state.status == 'run':
            break

    secret_votes = sorted(zip(last_proposal['team'], round_['mission']))
    for seat, vote in secret_votes:
        data.append({
            'game': game['id'],
            'player': game['players'][seat]['player_id'],
            'seat': seat,
            'role': hidden_state[seat],
            'is_evil': hidden_state[seat] in EVIL_ROLES,
            'type': 'mission',
            'move': vote,
            'propose_count': state.propose_count,
            'round': state.fails + state.succeeds,
            'succeeds': state.succeeds,
            'fails': state.fails,
            'propose_has_self': True,
            'num_players': len(hidden_state)
        })

    moves = [MissionAction(fail=(vote == "Fail")) for player, vote in secret_votes]
    state, _, _ = state.transition(moves, hidden_state)
    if state.status == 'merlin':
        assert 'findMerlin' in round_, "wat"
        find_merlin = round_['findMerlin']
        assert hidden_state[find_merlin['assassin']] == 'assassin', "wat"
        moves = [
            PickMerlinAction(merlin=find_merlin['merlin_guess'])
            for _ in hidden_state
        ]
        seat = hidden_state.index('assassin')
        data.append({
            'game': game['id'],
            'player': game['players'][seat]['player_id'],
            'seat': seat,
            'role': hidden_state[seat],
            'is_evil': hidden_state[seat] in EVIL_ROLES,
            'type': 'merlin',
            'move': str(find_merlin['merlin_guess']),
            'propose_count': state.propose_count,
            'round': state.fails + state.succeeds,
            'succeeds': state.succeeds,
            'fails': state.fails,
            'propose_has_self': True,
            'num_players': len(hidden_state)
        })
        state, _, _ = state.transition(moves, hidden_state)
    return state


def process_game(data, game):
    try:
        new_data = []
        hidden_state = reconstruct_hidden_state(game)
        print game['id']
        state = create_avalon_game(len(hidden_state)).start_state()
        for round_ in game['log']:
            state = handle_round(new_data, state, hidden_state, game, round_)
        data.extend(new_data)
    except AssertionError:
        print game['id'], 'is bad'


def get_data(games):
    data = []
    for game in games:
        process_game(data, game)
    return data



print "loading input json"
with open(DATAFILE, 'r') as f:
    input_json = json.load(f)


game_data = get_data(input_json)

ordered_cols = ['role', 'is_evil', 'type', 'move', 'propose_has_self',  'game', 'player', 'seat', 'propose_count', 'round', 'succeeds', 'fails', 'num_players']
dataframe = pd.DataFrame(game_data, columns=ordered_cols)
dataframe['type'] = dataframe['type'].astype('category')
dataframe['move'] = dataframe['move'].astype('category')
dataframe['role'] = dataframe['role'].astype('category')

print "Writing to msgpack"
with gzip.open("human_data_actions.msg.gz", 'w') as f:
    dataframe.to_msgpack(f)
