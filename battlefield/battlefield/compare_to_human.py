import numpy as np
import json
import os
import sys
import pandas as pd

from battlefield.avalon_types import GOOD_ROLES, EVIL_ROLES, possible_hidden_states, starting_hidden_states, ProposeAction, VoteAction, MissionAction, PickMerlinAction
from battlefield.avalon import AvalonState

DATAFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'bots', 'data', 'relabeled.json'))

ROLES = ['servant', 'merlin', 'percival', 'minion', 'assassin', 'mordred', 'morgana', 'oberon']

human_data = None

def reconstruct_hidden_state(game):
    roles = []
    for player in game['players']:
        roles.append('minion' if player['spy'] else 'servant')

    for role, p in game['roles'].items():
        if role not in set(['assassin', 'merlin', 'mordred', 'percival', 'morgana', 'oberon']):
            continue
        if roles[p] != 'assassin':
            roles[p] = role
    return tuple(roles)


def handle_transition(state, hidden_state, moves, bots):
    p_to_move = { p: move for p, move in zip(state.moving_players(), moves) }
    new_state, _, observation = state.transition(moves, hidden_state)
    for p, bot in enumerate(bots):
        bot.handle_transition(state, new_state, observation, move=p_to_move.get(p))
    return new_state


def get_prob(state, hidden_state, player, bot, move):
    legal_actions = state.legal_actions(player, hidden_state)
    move_probs = bot.get_move_probabilities(state, legal_actions)

    assert move in legal_actions, "Something is amiss"
    return move_probs[legal_actions.index(move)]


def handle_round(game, state, hidden_state, bots, round_, stats):
    last_proposal = None
    for proposal_num in ['1', '2', '3', '4', '5']:
        proposal = last_proposal = round_[proposal_num]
        assert state.proposer == proposal['proposer']
        assert state.propose_count == int(proposal_num) - 1
        moves = [ProposeAction(proposal=tuple(sorted(proposal['team'])))]
        for player, move in zip(state.moving_players(), moves):
            prob = get_prob(state, hidden_state, player, bots[player], move)
            stats.append({
                'game': game['id'],
                'seat': player,
                'role': hidden_state[player],
                'player': game['players'][player]['player_id'],
                'type': 'propose',
                'move': ','.join(map(str, sorted(move.proposal))),
                'bot': bots[player].__class__.__name__,
                'prob': prob,
                'num_players': len(hidden_state)
            })
        state = handle_transition(state, hidden_state, moves, bots)

        assert state.status == 'vote'
        moves = [VoteAction(up=(vote == 'Approve')) for vote in proposal['votes']]
        for player, move in zip(state.moving_players(), moves):
            prob = get_prob(state, hidden_state, player, bots[player], move)
            stats.append({
                'game': game['id'],
                'seat': player,
                'role': hidden_state[player],
                'player': game['players'][player]['player_id'],
                'type': 'vote',
                'move': 'up' if move.up else 'down',
                'bot': bots[player].__class__.__name__,
                'prob': prob,
                'num_players': len(hidden_state)
            })
        state = handle_transition(state, hidden_state, moves, bots)

        if state.status == 'run':
            break

    secret_votes = sorted(zip(last_proposal['team'], round_['mission']))
    moves = [MissionAction(fail=(vote == "Fail")) for player, vote in secret_votes]
    for player, move in zip(state.moving_players(), moves):
        prob = get_prob(state, hidden_state, player, bots[player], move)
        stats.append({
            'game': game['id'],
            'seat': player,
            'role': hidden_state[player],
            'player': game['players'][player]['player_id'],
            'type': 'mission',
            'move': 'fail' if move.fail else 'succeed',
            'bot': bots[player].__class__.__name__,
            'prob': prob,
            'num_players': len(hidden_state)
        })
    state = handle_transition(state, hidden_state, moves, bots)

    if state.status == 'merlin':
        assert 'findMerlin' in round_
        find_merlin = round_['findMerlin']
        assert hidden_state[find_merlin['assassin']] == 'assassin'
        moves = [
            PickMerlinAction(merlin=find_merlin['merlin_guess'])
            for _ in hidden_state
        ]
        for player, move in zip(state.moving_players(), moves):
            prob = get_prob(state, hidden_state, player, bots[player], move)
            if hidden_state[player] == 'assassin':
                stats.append({
                    'game': game['id'],
                    'seat': player,
                    'role': hidden_state[player],
                    'player': game['players'][player]['player_id'],
                    'type': 'merlin',
                    'move': move.merlin,
                    'bot': bots[player].__class__.__name__,
                    'prob': prob,
                    'num_players': len(hidden_state)
                })
        state = handle_transition(state, hidden_state, moves, bots)
    return state




def process_game(game, bot_class, stats, verbose=True, num_players=None, max_num_players=7, min_game_id=0, max_game_id=50000, roles=None):
    try:
        hidden_state = reconstruct_hidden_state(game)
        if num_players is not None:
            if len(hidden_state) != num_players:
                return
        else:
            if len(hidden_state) >= max_num_players:
                return

        if game['id'] >= max_game_id or game['id'] < min_game_id:
            return

        if roles is not None:
            if not all(role in roles for role in hidden_state):
                return

        if verbose:
            print game['id']
            print hidden_state
        possible = possible_hidden_states(set(hidden_state), num_players=len(hidden_state))
        perspectives = [
            starting_hidden_states(player, hidden_state, possible)
            for player, _ in enumerate(hidden_state)
        ]
        state = AvalonState.start_state(len(hidden_state))
        bots = [
            bot_class.create_and_reset(state, player, role, perspectives[player])
            for player, role in enumerate(hidden_state)
        ]
        for round_ in game['log']:
            state = handle_round(game, state, hidden_state, bots, round_, stats)
    except AssertionError:
        if verbose:
            print game['id'], 'is bad'


def create_stats():
    return []


def collect_stats(stats):
    d = pd.DataFrame(stats, columns=['game', 'bot', 'num_players', 'type', 'move', 'role', 'seat', 'player', 'prob'])
    d.bot = d.bot.astype('category')
    d.type = d.type.astype('category')
    d.move = d.move.astype('category')
    d.role = d.role.astype('category')
    return d


def load_human_data():
    global human_data
    if human_data is not None:
        return human_data
    print "Loading human data"
    sys.stdout.flush()
    with open(DATAFILE, 'r') as f:
        human_data = json.load(f)
    print "Done loading human"
    sys.stdout.flush()
    return human_data


def compute_human_statistics(bot_class, verbose=True, num_players=None, max_num_players=7, min_game_id=0, max_game_id=50000, roles=None):
    print "Analyzing {} from game {} to game {}".format(bot_class.__name__, min_game_id, max_game_id)
    stats = create_stats()
    sys.stdout.flush()
    data = load_human_data()
    for game in data:
        process_game(
            game,
            bot_class,
            stats,
            verbose=verbose,
            num_players=num_players,
            max_num_players=max_num_players,
            min_game_id=min_game_id,
            max_game_id=max_game_id,
            roles=roles
        )
    return collect_stats(stats)


def print_human_statistics(stats):
    bots = stats.bot.unique()
    for bot in bots:
        print "========== Bot statistics for {}".format(bot)
        data = stats[stats.bot == bot]
        print "All actions prob sum (doesnt make sense): {: >16.04f}".format(data.prob.sum())
        print "    Propose prob sum (doesnt make sense): {: >16.04f}".format(data[data.type == 'propose'].prob.sum())
        print "       Vote prob sum (doesnt make sense): {: >16.04f}".format(data[data.type == 'vote'].prob.sum())
        print "    Mission prob sum (doesnt make sense): {: >16.04f}".format(data[data.type == 'mission'].prob.sum())
        print "Pick Merlin prob sum (doesnt make sense): {: >16.04f}".format(data[data.type == 'merlin'].prob.sum())

