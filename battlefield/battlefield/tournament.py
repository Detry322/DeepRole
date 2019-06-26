import itertools
import pandas as pd
import multiprocessing
import gzip
import random
import os
from collections import defaultdict

from battlefield.avalon_types import GOOD_ROLES, EVIL_ROLES, possible_hidden_states, starting_hidden_states
from battlefield.avalon import AvalonState
import cPickle as pickle

def run_game(state, hidden_state, bots):
    print "game"
    while not state.is_terminal():
        moving_players = state.moving_players()
        moves = [
            bots[player].get_action(state, state.legal_actions(player, hidden_state))
            for player in moving_players
        ]
        new_state, _, observation = state.transition(moves, hidden_state)
        for player, bot in enumerate(bots):
            if player in moving_players:
                move = moves[moving_players.index(player)]
            else:
                move = None
            bot.handle_transition(state, new_state, observation, move=move)
        state = new_state

    return state.terminal_value(hidden_state), state.game_end



def run_large_tournament(bots_classes, roles, games_per_matching=50):
    print "Running {}".format(' '.join(map(lambda c: c.__name__, bots_classes)))

    start_state = AvalonState.start_state(len(roles))
    result = []
    all_hidden_states = possible_hidden_states(set(roles), num_players=len(roles))

    seen_hidden_states = set([])
    for hidden_state in itertools.permutations(roles):
        if hidden_state in seen_hidden_states:
            continue
        seen_hidden_states.add(hidden_state)

        beliefs = [
            starting_hidden_states(player, hidden_state, all_hidden_states) for player in range(len(hidden_state))
        ]
        seen_bot_orders = set([])
        for bot_order in itertools.permutations(bots_classes):
            bot_order_str = tuple([bot_cls.__name__ for bot_cls in bot_order])
            if bot_order_str in seen_bot_orders:
                continue
            seen_bot_orders.add(bot_order_str)

            for _ in range(games_per_matching):
                bots = [
                    bot_cls.create_and_reset(start_state, player, role, beliefs[player])
                    for player, (bot_cls, role) in enumerate(zip(bot_order, hidden_state))
                ]
                values, game_end = run_game(start_state, hidden_state, bots)
                game_stat = {
                    'winner': game_end[0],
                    'win_type': game_end[1],
                }
                for player, (bot_cls, role) in enumerate(zip(bot_order, hidden_state)):
                    game_stat['bot_{}'.format(player)] = bot_cls.__name__
                    game_stat['bot_{}_role'.format(player)] = role
                    game_stat['bot_{}_payoff'.format(player)] = values[player]
                result.append(game_stat)

    df = pd.DataFrame(result, columns=sorted(result[0].keys()))
    df['winner'] = df['winner'].astype('category')
    df['win_type'] = df['win_type'].astype('category')
    for player in range(len(roles)):
        df['bot_{}'.format(player)] = df['bot_{}'.format(player)].astype('category')
        df['bot_{}_role'.format(player)] = df['bot_{}_role'.format(player)].astype('category')

    return df


def large_tournament_parallel_helper(bot_order, hidden_state, beliefs, start_state):
    bots = [
        bot_cls.create_and_reset(start_state, player, role, beliefs[player])
        for player, (bot_cls, role) in enumerate(zip(bot_order, hidden_state))
    ]
    values, game_end = run_game(start_state, hidden_state, bots)
    game_stat = {
        'winner': game_end[0],
        'win_type': game_end[1],
    }
    for player, (bot_cls, role) in enumerate(zip(bot_order, hidden_state)):
        game_stat['bot_{}'.format(player)] = bot_cls.__name__
        game_stat['bot_{}_role'.format(player)] = role
        game_stat['bot_{}_payoff'.format(player)] = values[player]

    return game_stat


def run_large_tournament_parallel(pool, bots_classes, roles, games_per_matching=50):
    print "Running {}".format(' '.join(map(lambda c: c.__name__, bots_classes)))

    start_state = AvalonState.start_state(len(roles))
    async_results = []
    all_hidden_states = possible_hidden_states(set(roles), num_players=len(roles))

    seen_hidden_states = set([])
    for hidden_state in itertools.permutations(roles):
        if hidden_state in seen_hidden_states:
            continue
        seen_hidden_states.add(hidden_state)

        beliefs = [
            starting_hidden_states(player, hidden_state, all_hidden_states) for player in range(len(hidden_state))
        ]
        seen_bot_orders = set([])
        for bot_order in itertools.permutations(bots_classes):
            bot_order_str = tuple([bot_cls.__name__ for bot_cls in bot_order])
            if bot_order_str in seen_bot_orders:
                continue
            seen_bot_orders.add(bot_order_str)

            for _ in range(games_per_matching):
                async_result = pool.apply_async(large_tournament_parallel_helper, (bot_order, hidden_state, beliefs, start_state))
                async_results.append(async_result)
                
    result = [ async_result.get() for async_result in async_results ]

    df = pd.DataFrame(result, columns=sorted(result[0].keys()))
    df['winner'] = df['winner'].astype('category')
    df['win_type'] = df['win_type'].astype('category')
    for player in range(len(roles)):
        df['bot_{}'.format(player)] = df['bot_{}'.format(player)].astype('category')
        df['bot_{}_role'.format(player)] = df['bot_{}_role'.format(player)].astype('category')

    return df


def run_game_and_create_bots(hidden_state, beliefs, config):
    start_state = AvalonState.start_state(len(hidden_state))
    bots = [ bot['bot']() for bot in config ]
    for player, (bot, config) in enumerate(zip(bots, config)):
        bot.reset(start_state, player, config['role'], beliefs[player])
    return run_game(start_state, hidden_state, bots)


def run_simple_tournament(config, num_games=1000, granularity=100):
    tournament_statistics = {
        'bots': [
            { 'bot': bot['bot'].__name__, 'role': bot['role'], 'wins': 0, 'total': 0, 'win_percent': 0, 'payoff': 0.0 }
            for bot in config
        ],
        'end_types': {}
    }


    hidden_state = tuple([bot['role'] for bot in config])
    all_hidden_states = possible_hidden_states(set(hidden_state), num_players=len(config))
    beliefs = [
        starting_hidden_states(player, hidden_state, all_hidden_states) for player in range(len(config))
    ]

    pool = multiprocessing.Pool(4)
    results = []

    for i in range(num_games):
        results.append(pool.apply_async(run_game_and_create_bots, (hidden_state, beliefs, config)))

    for i, result in enumerate(results):
        if i % granularity == 0:
            print "Waiting for game {}".format(i)
        payoffs, end_type = result.get()
        tournament_statistics['end_types'][end_type] = 1 + tournament_statistics['end_types'].get(end_type, 0)

        for b, payoff in zip(tournament_statistics['bots'], payoffs):
            b['wins'] += 1 if payoff > 0.0 else 0
            b['payoff'] += payoff
            b['total'] += 1

    for b in tournament_statistics['bots']:
        b['win_percent'] = float(b['wins'])/float(b['total'])

    pool.close()
    pool.join()

    return tournament_statistics


def check_config(config):
    role_counts = defaultdict(lambda: 0)

    start_state = AvalonState.start_state(len(config))

    for bot in config:
        # Count roles
        role_counts[bot['role']] += 1

    assert 'merlin' in role_counts
    assert 'assassin' in role_counts
    for role, count in role_counts.items():
        assert role == 'servant' or role == 'minion' or count == 1
        assert role in GOOD_ROLES or role in EVIL_ROLES


def print_tournament_statistics(tournament_statistics):
    print "       Role |                      Bot | Evil |      Winrate |        Payoff "
    print "-----------------------------------------------------------------------------"
    for bot in tournament_statistics['bots']:
        print "{: >11} | {: >24} | {: >4} | {: >11.02f}% | {: >13.02f} ".format(bot['role'], bot['bot'], 'Yes' if bot['role'] in EVIL_ROLES else '', 100*bot['win_percent'], bot['payoff'])

    for game_end, count in sorted(tournament_statistics['end_types'].items(), key=lambda x: x[1], reverse=True):
        print "{}: {} - {}".format(count, game_end[0], game_end[1])



def run_all_combos_simple(bots, roles, games_per_matching=50):
    results = []
    for combination in itertools.combinations_with_replacement(bots, r=len(roles)):
        combo_name = '-'.join(map(lambda c: c.__name__, combination))
        results.append(
            (combo_name, run_large_tournament(combination, roles, games_per_matching=games_per_matching))
        )

    job_id = os.urandom(10).encode('hex')

    for combo_name, dataframe in results:
        filename = 'tournaments/{}_{}.msg.gz'.format(combo_name, job_id)
        print "Writing {}".format(filename)
        with gzip.open(filename, 'w') as f:
            dataframe.to_msgpack(f)


def run_all_combos(bots, roles, games_per_matching=50, parallelization=16):
    pool = multiprocessing.Pool(parallelization)
    results = []
    job_id = os.urandom(10).encode('hex')
    for combination in itertools.combinations_with_replacement(bots, r=len(roles)):
        combo_name = '-'.join(map(lambda c: c.__name__, combination))
        result = run_large_tournament_parallel(pool, combination, roles, games_per_matching=games_per_matching)
        filename = 'tournaments/{}_{}.msg.gz'.format(combo_name, job_id)
        print "Writing {}".format(filename)
        with gzip.open(filename, 'w') as f:
            dataframe.to_msgpack(f)

    pool.close()
    pool.join()



def run_all_combos_parallel(bots, roles):
    pool = multiprocessing.Pool()
    results = []

    for combination in itertools.combinations_with_replacement(bots, r=len(roles)):
        combo_name = '-'.join(map(lambda c: c.__name__, combination))
        results.append(
            (combo_name, pool.apply_async(run_large_tournament, (combination, roles)))
        )

    for combo_name, async_result in results:
        dataframe = async_result.get()
        filename = 'tournaments/{}.msg.gz'.format(combo_name)
        print "Writing {}".format(filename)
        with gzip.open(filename, 'w') as f:
            dataframe.to_msgpack(f)




def run_learning_tournament(bot_classes, winrate_track=None, winrate_window=10000000):
    bots = [
        ( bot_class(), bot_class.__name__, num == winrate_track )
        for num, bot_class in enumerate(bot_classes)
    ]

    hidden_state = ['merlin', 'servant', 'servant', 'assassin', 'minion']
    all_hidden_states = possible_hidden_states(set(hidden_state), num_players=len(hidden_state))

    beliefs_for_hidden_state = {}
    start_state = AvalonState.start_state(len(hidden_state))

    wins = []
    game_num = 0

    while True:
        game_num += 1
        random.shuffle(hidden_state)
        random.shuffle(bots)
        bot_ids = [ bot_name for _, bot_name, _ in bots ]

        if tuple(hidden_state) not in beliefs_for_hidden_state:
            beliefs_for_hidden_state[tuple(hidden_state)] = [
                starting_hidden_states(
                    player,
                    tuple(hidden_state),
                    all_hidden_states
                ) for player in range(len(hidden_state))
            ]

        beliefs = beliefs_for_hidden_state[tuple(hidden_state)]

        track_num = None

        bot_objs = []
        for i, (bot, bot_name, track) in enumerate(bots):
            if track:
                track_num = i
            bot.reset(start_state, i, hidden_state[i], beliefs[i])
            bot.set_bot_ids(bot_ids)
            bot_objs.append(bot)

        results, _ = run_game(start_state, tuple(hidden_state), bot_objs)

        for i, (bot, bot_name, track) in enumerate(bots):
            bot.show_roles(hidden_state, bot_ids)

        if track_num is not None:
            wins.append(int(results[track_num] > 0))
            if len(wins) > winrate_window:
                wins.pop(0)

        if game_num % 10 == 0 and winrate_track is not None:
            print "Winrate: {}%".format(100 * float(sum(wins)) / len(wins))


def run_single_threaded_tournament(config, num_games=1000, granularity=100):
    tournament_statistics = {
        'bots': [
            { 'bot': bot['bot'].__name__, 'role': bot['role'], 'wins': 0, 'total': 0, 'win_percent': 0, 'payoff': 0.0 }
            for bot in config
        ],
        'end_types': {}
    }


    hidden_state = tuple([bot['role'] for bot in config])
    all_hidden_states = possible_hidden_states(set(hidden_state), num_players=len(config))
    beliefs = [
        starting_hidden_states(player, hidden_state, all_hidden_states) for player in range(len(config))
    ]

    start_state = AvalonState.start_state(len(hidden_state))
    bots = [ bot['bot']() for bot in config ]

    # pool = multiprocessing.Pool()
    results = []

    for i in range(num_games):
        if i % granularity == 0:
            print i
        for player, (bot, c) in enumerate(zip(bots, config)):
            bot.reset(start_state, player, c['role'], beliefs[player])

        payoffs, end_type = run_game(start_state, hidden_state, bots)
        tournament_statistics['end_types'][end_type] = 1 + tournament_statistics['end_types'].get(end_type, 0)

        for b, payoff in zip(tournament_statistics['bots'], payoffs):
            b['wins'] += 1 if payoff > 0.0 else 0
            b['payoff'] += payoff
            b['total'] += 1

    for b in tournament_statistics['bots']:
        b['win_percent'] = float(b['wins'])/float(b['total'])

    return tournament_statistics


def run_and_print_game(config):
    bot_counts = {}
    bot_names = []
    for bot in config:
        base_name = bot['bot'].__name__
        bot_counts[base_name] = bot_counts.get(base_name, 0) + 1
        bot_names.append("{}_{}".format(base_name, bot_counts[base_name]))

    print "       Role |                      Bot | Evil "
    print "----------------------------------------------"
    for name, bconf in zip(bot_names, config):
        print "{: >11} | {: >24} | {: >4}".format(bconf['role'], name, 'Yes' if bconf['role'] in EVIL_ROLES else '')

    hidden_state = tuple([bot['role'] for bot in config])
    all_hidden_states = possible_hidden_states(set(hidden_state), num_players=len(config))
    beliefs = [
        starting_hidden_states(player, hidden_state, all_hidden_states) for player in range(len(config))
    ]
    state = AvalonState.start_state(len(hidden_state))
    bots = [ bot['bot']() for bot in config ]
    for i, bot in enumerate(bots):
        bot.reset(state, i, hidden_state[i], beliefs[i])

    print "=============== Round 1 ================"
    while not state.is_terminal():

        moving_players = state.moving_players()
        moves = [
            bots[player].get_action(state, state.legal_actions(player, hidden_state))
            for player in moving_players
        ]
        if state.status == 'propose':
            player = moving_players[0]
            legal_actions = state.legal_actions(player, hidden_state)
            move_probs = bots[player].get_move_probabilities(state, legal_actions)
            move_prob = move_probs[legal_actions.index(moves[0])]
            print "Proposal #{}. {} proposes ({:0.2f}):".format(state.propose_count + 1, bot_names[moving_players[0]], move_prob)
            for player in moves[0].proposal:
                print " - {}".format(bot_names[player])
        elif state.status == 'vote':
            for player, move in zip(moving_players, moves):
                legal_actions = state.legal_actions(player, hidden_state)
                move_probs = bots[player].get_move_probabilities(state, legal_actions)
                move_prob = move_probs[legal_actions.index(move)]
                print "{: >24} votes {: <4} ({:0.2f})".format(bot_names[player], 'UP' if move.up else 'DOWN', move_prob)
        elif state.status == 'run':
            print "--- Mission results ---"
            for player, move in zip(moving_players, moves):
                legal_actions = state.legal_actions(player, hidden_state)
                move_probs = bots[player].get_move_probabilities(state, legal_actions)
                move_prob = move_probs[legal_actions.index(move)]
                print "{: >24}: {} ({:0.2f})".format(bot_names[player], 'FAIL' if move.fail else 'SUCCEED', move_prob)
        elif state.status == 'merlin':
            print "===== Final chance: pick merlin! ====="
            assassin = hidden_state.index('assassin')
            legal_actions = state.legal_actions(assassin, hidden_state)
            move_probs = bots[assassin].get_move_probabilities(state, legal_actions)
            move_prob = move_probs[legal_actions.index(moves[0])]
            assassin_pick = moves[assassin].merlin
            print '{} picked {} - {}! ({:0.2f})'.format(bot_names[assassin], bot_names[assassin_pick], 'CORRECT' if assassin_pick == hidden_state.index('merlin') else 'WRONG', move_prob)

        new_state, _, observation = state.transition(moves, hidden_state)
        for player, bot in enumerate(bots):
            if player in moving_players:
                move = moves[moving_players.index(player)]
            else:
                move = None
            bot.handle_transition(state, new_state, observation, move=move)

        if state.status == 'run' and new_state.status == 'propose':
            print "=============== Round {} ================".format(new_state.succeeds + new_state.fails + 1)

        state = new_state

    print state.game_end
