import os
import gzip
import cPickle as pickle
import json
import sys

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.append(PARENT_DIR)

from battlefield.avalon import AvalonState
from battlefield.avalon_types import EVIL_ROLES, GOOD_ROLES, ProposeAction, VoteAction, MissionAction, PickMerlinAction, possible_hidden_states, starting_hidden_states

DATAFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'relabeled.json'))
OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data.pkl.gz'))

print "loading input json"
with open(DATAFILE, 'r') as f:
    input_json = json.load(f)


def perspective_from_hidden_states(hidden_states):
    roles = [set([]) for _ in range(len(hidden_states[0]))]
    for hidden_state in hidden_states:
        for p, role in enumerate(hidden_state):
            roles[p].add(role)
    return tuple([ frozenset(possible) for possible in roles ])


def reconstruct_hidden_state(game):
    roles = []
    for player in game['players']:
        roles.append('minion' if player['spy'] else 'servant')

    for role, p in game['roles'].items():
        if role not in set(['assassin', 'merlin', 'mordred', 'percival', 'morgana', 'oberon']):
            continue
        roles[p] = role
    return tuple(roles)


def deal_with_transition(tree_roots, state, moves, hidden_state):
    new_state, _, observation = state.transition(moves, hidden_state)
    move_map = {
        p: m
        for p, m in zip(state.moving_players(), moves)
    }
    for p, m in move_map.items():
        tree_roots[p]['move_counts'][m] = 1 + tree_roots[p]['move_counts'].get(m, 0)

    new_roots = [
        root['transitions'].setdefault(
            (new_state.as_key(), observation, move_map.get(p)),
            { 'move_counts': {}, 'transitions': {} }
        )
        for p, root in enumerate(tree_roots)
    ]
    return new_roots, new_state




def handle_round(tree_roots, state, hidden_state, round_):
    last_proposal = None
    for proposal_num in ['1', '2', '3', '4', '5']:
        proposal = last_proposal = round_[proposal_num]
        assert state.proposer == proposal['proposer'], "idk"
        assert state.propose_count == int(proposal_num) - 1, "idk2"
        moves = [ProposeAction(proposal=tuple(sorted(proposal['team'])))]
        tree_roots, state = deal_with_transition(tree_roots, state, moves, hidden_state)
        assert state.status == 'vote'
        moves = [VoteAction(up=(vote == 'Approve')) for vote in proposal['votes']]
        tree_roots, state = deal_with_transition(tree_roots, state, moves, hidden_state)
        if state.status == 'run':
            break

    secret_votes = sorted(zip(last_proposal['team'], round_['mission']))
    moves = [MissionAction(fail=(vote == "Fail")) for player, vote in secret_votes]
    tree_roots, state = deal_with_transition(tree_roots, state, moves, hidden_state)
    if state.status == 'merlin':
        assert 'findMerlin' in round_, "wat"
        find_merlin = round_['findMerlin']
        assert hidden_state[find_merlin['assassin']] == 'assassin', "wat"
        moves = [
            PickMerlinAction(merlin=find_merlin['merlin_guess'])
            for _ in hidden_state
        ]
        tree_roots, state = deal_with_transition(tree_roots, state, moves, hidden_state)
    return tree_roots, state




def process_game(root, game):
    try:
        hidden_state = reconstruct_hidden_state(game)
        if len(hidden_state) >= 7:
            return
        print game['id']
        possible = possible_hidden_states(set(hidden_state), num_players=len(hidden_state))
        perspectives = [
            perspective_from_hidden_states(starting_hidden_states(player, hidden_state, possible))
            for player, _ in enumerate(hidden_state)
        ]
        tree_roots = [
            root.setdefault((player, perspective), { 'move_counts': {}, 'transitions': {}})
            for player, perspective in enumerate(perspectives)
        ]
        state = AvalonState.start_state(len(hidden_state))
        for round_ in game['log']:
            tree_roots, state = handle_round(tree_roots, state, hidden_state, round_)
    except AssertionError:
        print game['id'], 'is bad'


game_tree_root = {}

for game in input_json:
    process_game(game_tree_root, game)

with gzip.open(OUTPUT, 'w') as f:
    pickle.dump(game_tree_root, f)
