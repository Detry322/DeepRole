import pandas as pd
import numpy as np
import copy
import gzip
import itertools
import multiprocessing
import sys
import os
import glob
import json
import cPickle as pickle

from battlefield.avalon_types import (
    GOOD_ROLES, EVIL_ROLES, possible_hidden_states, starting_hidden_states,
    ProposeAction, VoteAction, MissionAction, PickMerlinAction
)
from battlefield.avalon import AvalonState
from battlefield.bots.deeprole import Deeprole

HUMAN_DATA_BASE = os.path.join(
    os.path.abspath(os.path.dirname(__file__)),
    '..',
    '..',
    'proavalon'
)

def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)


def load_all_games():
    games = {}
    for game_file in glob.glob(os.path.join(HUMAN_DATA_BASE, 'results_dir/*/*/*.json')):
        with open(game_file) as f:
            g = json.load(f)

            key = (
                tuple(g['session_info']['players']),
                g['game_info']['roomId'],
                tuple(g['game_info']['roles']),
                g['game_info']['missionNum'],
                g['game_info']['teamLeader'],
                tuple(g['game_info']['proposedTeam']),
                g['game_info']['winner'],
                to_tuple(g['game_info']['voteHistory'][g['session_info']['players'][0]]),
                to_tuple(g['game_info']['voteHistory'][g['session_info']['players'][1]]),
                to_tuple(g['game_info']['voteHistory'][g['session_info']['players'][2]]),
                to_tuple(g['game_info']['voteHistory'][g['session_info']['players'][3]]),
                to_tuple(g['game_info']['voteHistory'][g['session_info']['players'][4]]),
            )
            games[key] = g
    # with open('/Users/jserrino/playground/translate/readable_records_assassin.json') as f:
    #     return json.load(f)
    return games


PRO_TO_HS = {
    'Resistance': 'servant',
    'Spy': 'minion',
    'Assassin': 'assassin',
    'Merlin': 'merlin'
}

def replay_game(game):
    roles = game['game_info']['roles']
    roles = roles[:1] + roles[1:][::-1]
    hidden_state = [PRO_TO_HS[r] for r in roles]
    players = game['session_info']['players']

    proposer = [
        'VHleader' in game['game_info']['voteHistory'][player][0][0]
        for player in players
    ].index(True)

    state = AvalonState(
        proposer=proposer,
        propose_count=0,
        succeeds=0,
        fails=0,
        status='propose',
        proposal=None,
        game_end=None,
        num_players=5
    )

    yield None, state, hidden_state

    while not state.is_terminal():
        rnd = state.succeeds + state.fails
        if state.status != 'merlin':
            proposer = [
                'VHleader' in game['game_info']['voteHistory'][player][rnd][state.propose_count]
                for player in players
            ].index(True)
            assert proposer == state.proposer, "{} != {}".format(proposer, state.proposer)
        if state.status == 'propose':
            proposal = tuple(sorted([
                players.index(player)
                for player in players
                if 'VHpicked' in game['game_info']['voteHistory'][player][rnd][state.propose_count]
            ]))
            actions = [ProposeAction(proposal=proposal)]
        elif state.status == 'vote':
            actions = [
                VoteAction(up=(
                    'VHapprove' in game['game_info']['voteHistory'][player][rnd][state.propose_count]
                ))
                for player in players
            ]
        elif state.status == 'run':
            observed_fails = game['game_info']['numFailsHistory'][rnd]
            actions = []
            for player in state.moving_players():
                if hidden_state[player] in set(['merlin', 'servant']):
                    actions.append(MissionAction(fail=False))
                elif observed_fails == 0:
                    actions.append(MissionAction(fail=False))
                else:
                    actions.append(MissionAction(fail=True))
                    observed_fails -= 1
            assert observed_fails == 0
        elif state.status == 'merlin':
            shot_player = players.index(game['game_info']['publicData']['roles']['assassinShotUsername'])
            actions = [PickMerlinAction(merlin=shot_player) for _ in range(5)]

        assert len(actions) == len(state.moving_players())

        new_state, _, obs = state.transition(actions, hidden_state)
        yield state, new_state, obs
        state = new_state


def deeprole_is_res_and_4_humans(g):
    roles = g['game_info']['roles']
    roles = roles[:1] + roles[1:][::-1]
    players = g['session_info']['players']
    if len([ p for p in players if 'DeepRole#' in p ]) != 1:
        return False

    deeprole_seat = ['DeepRole#' in p for p in players].index(True)
    if roles[deeprole_seat] != 'Resistance':
        return False

    if 'Assassin' not in roles or 'Merlin' not in roles:
        return False

    return (True, deeprole_seat)


def deeprole_is_assassin_and_4_humans(g):
    roles = g['game_info']['roles']
    roles = roles[:1] + roles[1:][::-1]
    players = g['session_info']['players']
    if len([ p for p in players if 'DeepRole#' in p ]) != 1:
        return False

    deeprole_seat = ['DeepRole#' in p for p in players].index(True)
    if roles[deeprole_seat] != 'Assassin':
        return False

    if 'Assassin' not in roles or 'Merlin' not in roles:
        return False

    return (True, deeprole_seat)


def human_is_merlin_and_4_bots(g):
    roles = g['game_info']['roles']
    roles = roles[:1] + roles[1:][::-1]
    players = g['session_info']['players']
    if len([ p for p in players if 'DeepRole#' in p ]) != 4:
        return False

    if len(players) != 5:
        return False

    human_seat = ['DeepRole#' not in p for p in players].index(True)
    if roles[human_seat] != 'Merlin':
        return False

    if 'Assassin' not in roles or 'Merlin' not in roles:
        return False

    return (True, 0 if human_seat != 0 else 1)


def recover_deeprole_thoughts(game):
    _, seat = human_is_merlin_and_4_bots(game)
    iterator = replay_game(game)
    _, state, hidden_state = next(iterator)
    print game['session_info']['players']

    bot = Deeprole.create_and_reset(state, seat, hidden_state[seat], [hidden_state])

    results = []
    for state, new_state, obs in iterator:
        results.append((state.succeeds, state.fails, state.propose_count, bot.belief))
        bot.handle_transition(state, new_state, obs)

    return {
        'log': results,
        'winner': game['game_info']['winner'],
        'game_end': new_state.game_end,
        'hidden_state': hidden_state,
        'seat': seat,
    }


def recover_all_deeprole_thoughts():
    print "loading games"
    games = load_all_games().values()
    print "total", len(games)
    print "filtering games"
    filtered = [ g for g in games if human_is_merlin_and_4_bots(g) ]
    print "total", len(filtered)

    for i, game in enumerate(filtered):
        print i, "of", len(filtered)
        try:
            thoughts = recover_deeprole_thoughts(game)
            with open('bot_guess_merlin/{}.pkl'.format(os.urandom(5).encode('hex')), 'w') as f:
                pickle.dump(thoughts, f)
        except KeyboardInterrupt:
            exit(0)
        except:
            print i, "errored"
            pass
