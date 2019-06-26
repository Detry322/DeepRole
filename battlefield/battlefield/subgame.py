import itertools
import random
import numpy as np
import warnings
from collections import defaultdict

from battlefield.avalon_types import GOOD_ROLES, EVIL_ROLES, possible_hidden_states, starting_hidden_states, ProposeAction, VoteAction, MissionAction, PickMerlinAction
from battlefield.avalon import AvalonState
from battlefield.bots import SimpleStatsBot, ObserveBot, RandomBot, HumanLikeBot
from battlefield.bots.observe_beater_bot import get_python_perspective


def calculate_observation_ll(hidden_state, bot_classes, observation_history, tremble=0.0):
    all_hidden_states = possible_hidden_states(set(hidden_state), num_players=len(hidden_state))
    beliefs = [
        starting_hidden_states(player, hidden_state, all_hidden_states) for player in range(len(hidden_state))
    ]
    state = AvalonState.start_state(len(hidden_state))
    bots = [ bot() for bot in bot_classes ]
    for i, bot in enumerate(bots):
        bot.reset(state, i, hidden_state[i], beliefs[i])

    log_likelihood = 0.0

    for obs_type, obs in observation_history:
        assert obs_type == state.status, "Incorrect matchup {} != {}".format(obs_type, state.status)

        moving_players = state.moving_players()
        moves = []

        if obs_type == 'propose':
            player = moving_players[0]
            legal_actions = state.legal_actions(player, hidden_state)
            move = ProposeAction(proposal=obs)
            index = legal_actions.index(move)
            moves.append(move)
            move_probs = bots[player].get_move_probabilities(state, legal_actions)
            move_probs = (1.0 - tremble) * move_probs + tremble * (np.ones(len(legal_actions))/len(legal_actions))
            log_likelihood += np.log(move_probs[index])
        elif obs_type == 'vote':
            for p, vote_up in zip(moving_players, obs):
                legal_actions = state.legal_actions(p, hidden_state)
                move = VoteAction(up=vote_up)
                index = legal_actions.index(move)
                moves.append(move)
                move_probs = bots[p].get_move_probabilities(state, legal_actions)
                move_probs = (1.0 - tremble) * move_probs + tremble * (np.ones(len(legal_actions))/len(legal_actions))
                log_likelihood += np.log(move_probs[index])
        elif obs_type == 'run':
            bad_guys_on_mission = [p for p in state.proposal if hidden_state[p] in EVIL_ROLES ]
            if len(bad_guys_on_mission) < obs:
                # Impossible - fewer bad than failed
                return np.log(0.0)

            player_fail_probability = {}
            for bad in bad_guys_on_mission:
                legal_actions = state.legal_actions(bad, hidden_state)
                move = MissionAction(fail=True)
                index = legal_actions.index(move)
                move_probs = bots[bad].get_move_probabilities(state, legal_actions)
                move_probs = (1.0 - tremble) * move_probs + tremble * (np.ones(len(legal_actions))/len(legal_actions))
                player_fail_probability[bad] = move_probs[index]


            failure_prob = 0.0
            moves = [ MissionAction(fail=False) ] * len(state.proposal)
            for bad_failers in itertools.combinations(bad_guys_on_mission, r=obs):
                specific_fail_prob = 1.0
                for bad in bad_guys_on_mission:
                    moves[state.proposal.index(bad)] = MissionAction(fail=True) if bad in bad_failers else MissionAction(fail=False)
                    specific_fail_prob *= player_fail_probability[bad] if bad in bad_failers else (1.0 - player_fail_probability[bad])
                failure_prob += specific_fail_prob
            log_likelihood += np.log(failure_prob)

        new_state, _, observation = state.transition(moves, hidden_state)
        for player, bot in enumerate(bots):
            if player in moving_players:
                move = moves[moving_players.index(player)]
            else:
                move = None
            bot.handle_transition(state, new_state, observation, move=move)
        state = new_state
    return log_likelihood


def calculate_subgame_ll(roles, num_players, bot_classes, observation_history, tremble=0.0):
    hidden_states = possible_hidden_states(roles, num_players)
    ll = np.zeros(len(hidden_states))

    for h, hidden_state in enumerate(hidden_states):
        with np.errstate(divide='ignore'):
            ll[h] = calculate_observation_ll(hidden_state, bot_classes, observation_history, tremble=tremble)

    return hidden_states, ll


UNIFORM_STRATS = {
    2: np.ones(2) / 2.0,
    5: np.ones(5) / 5.0,
    10: np.ones(10) / 10.0
}

def calculate_strategy(regrets):
    reg = np.clip(regrets, 0, None)
    s = np.sum(reg)
    if s == 0:
        return UNIFORM_STRATS[len(regrets)]
    reg /= s
    return reg


def get_action_index(strat):
    target = np.random.random()
    for i in range(len(strat)):
        if target < strat[i]:
            return i
        target -= strat[i]
    return len(strat) - 1


def subgame_cfr(state, hidden_state, perspectives, me, regrets, strats, observations, strategy_probability, t):
    if state.is_terminal():
        return state.terminal_value(hidden_state)[me]

    observation_history = tuple(observations)

    moving_players = state.moving_players()
    my_move_index = None
    moves = [None] * len(moving_players)
    for i, player in enumerate(moving_players):
        if hidden_state[player] not in EVIL_ROLES and state.status == 'run':
            moves[i] = MissionAction(fail=False)
            continue

        if hidden_state[player] != 'assassin' and state.status == 'merlin':
            moves[i] = PickMerlinAction(merlin=np.random.choice(len(hidden_state)))
            continue

        perspective = perspectives[player]

        if player == me:
            my_move_index = i
            continue
        
        move_probs = calculate_strategy(regrets[state.status][(perspective, observation_history)])
        legal_actions = state.legal_actions(player, hidden_state)
        moves[i] = legal_actions[get_action_index(move_probs)]


    if my_move_index is None:
        new_state, _, observation = state.transition(moves, hidden_state)
        if state.status == 'vote':
            observation = tuple([vote.up for vote in observation])
        observations.append(observation)
        value = subgame_cfr(new_state, hidden_state, perspectives, me, regrets, strats, observations, strategy_probability, t)
        observations.pop()
        return value


    perspective = perspectives[me]
    p = calculate_strategy(regrets[state.status][(perspective, observation_history)])

    values = np.zeros(len(p))

    legal_actions = state.legal_actions(me, hidden_state)
    for action_index in range(len(values)):
        moves[my_move_index] = legal_actions[action_index]
        new_state, _, observation = state.transition(moves, hidden_state)
        if state.status == 'vote':
            observation = tuple([vote.up for vote in observation])
        observations.append(observation)
        values[action_index] = subgame_cfr(new_state, hidden_state, perspectives, me, regrets, strats, observations, strategy_probability * p[action_index], t)
        observations.pop()

    strategy_value = np.dot(values, p)
    new_regrets = values - strategy_value
    key = (perspective, observation_history)
    regrets[state.status][key] += new_regrets * t
    strats[state.status][key] += p * strategy_probability * t
    return strategy_value


def get_player_values(hidden_state, state, perspectives, observations, strats, probability):
    if probability < 0.000000001:
        return np.zeros(len(hidden_state))

    if state.is_terminal():
        return probability * np.array(state.terminal_value(hidden_state))

    observation_history = tuple(observations)
    moving_players = state.moving_players()

    unnormalized_strats = [
        (
            np.array([1.0])
            if hidden_state[player] not in EVIL_ROLES and state.status == 'run' else
            np.array([1.0, 0, 0, 0, 0])
            if hidden_state[player] != 'assassin' and state.status == 'merlin' else
            strats[state.status][(perspectives[player], observation_history)]
        ) for player in moving_players
    ]
    normalized_strats = [ strat / np.sum(strat) if np.sum(strat) != 0 else np.ones(len(strat)) / len(strat) for strat in unnormalized_strats ]
    legal_actions = [ state.legal_actions(player, hidden_state) for player in moving_players ]
    
    values = np.zeros(len(hidden_state))

    transition_cache = None
    for move_indices in itertools.product(*[range(len(player_strat)) for player_strat in normalized_strats]):
        moves = [actions[index] for actions, index in zip(legal_actions, move_indices)]
        p = 1.0
        for strat, index in zip(normalized_strats, move_indices):
            p *= strat[index]
            if p == 0:
                break
        if p == 0.0:
            continue

        if state.status == 'vote' and transition_cache is not None:
            new_state = transition_cache
            observation = moves
        else:
            new_state, _, observation = state.transition(moves, hidden_state)

        if transition_cache is None:
            transition_cache = new_state

        if state.status == 'vote':
            observation = tuple([vote.up for vote in observation])
        observations.append(observation)
        values += get_player_values(hidden_state, new_state, perspectives, observations, strats, p * probability)
        observations.pop()
    return values


def get_player_values_for_state(hidden_states, probs, state, strats):
    player_values_in_state = np.zeros((len(hidden_states), len(hidden_states[0])))

    for h, hidden_state in enumerate(hidden_states):
        print "player values", h
        if probs[h] == 0:
            continue
        perspectives = [
            get_python_perspective(hidden_state, player) for player in range(5)
        ]
        player_values_in_state[h] = get_player_values(hidden_state, state, perspectives, [], strats, 1.0)
    return player_values_in_state


def get_counterfactual_value(player, hidden_states_by_index, player_values_in_state, probs):
    value = 0.0
    prob_sum = 0.0
    for index in hidden_states_by_index:
        value += player_values_in_state[index][player] * probs[index]
        prob_sum += probs[index]

    return value / prob_sum


def get_counterfactual_value_by_perspective(perspective, hidden_states, player_values_in_state, probs):
    _, player, _ = perspective
    indices = []
    for h, hidden_state in enumerate(hidden_states):
        if get_python_perspective(hidden_state, player) == perspective:
            indices.append(h)
    return get_counterfactual_value(player, indices, player_values_in_state, probs)


def solve_subgame(hidden_states, lls, state, iterations=1000):
    assert state.succeeds == 2 and state.fails == 2 and state.propose_count > 2
    lls = lls - np.max(lls)
    probs = np.exp(lls)
    probs /= np.sum(probs)

    regrets = {
        'propose': defaultdict(lambda: np.zeros(10)),
        'vote': defaultdict(lambda: np.zeros(2)),
        'run': defaultdict(lambda: np.zeros(2)),
        'merlin': defaultdict(lambda: np.zeros(5)),
    }

    strats = {
        'propose': defaultdict(lambda: np.zeros(10)),
        'vote': defaultdict(lambda: np.zeros(2)),
        'run': defaultdict(lambda: np.zeros(2)),
        'merlin': defaultdict(lambda: np.zeros(5)),
    }

    for t in range(iterations):
        if t % 100 == 0:
            print t
        h = np.random.choice(len(hidden_states), p=probs)
        hidden_state = hidden_states[h]
        perspectives = [
            get_python_perspective(hidden_state, player) for player in range(5)
        ]
        for player in range(5):
            subgame_cfr(state, hidden_state, perspectives, player, regrets, strats, [], 1.0, t + 1.0)

    print "Retreiving player values..."
    player_values_in_state = get_player_values_for_state(hidden_states, probs, state, strats)
    interesting_perspective = ('player', 3, None)
    print "Player 3's perspective (if servant): {}".format(
        get_counterfactual_value_by_perspective(
            interesting_perspective,
            hidden_states,
            player_values_in_state,
            probs
        )
    )



def test_calculate():
    roles = ['merlin', 'assassin', 'minion', 'servant']
    # bot_classes = [ RandomBot, RandomBot, ObserveBot, RandomBot, RandomBot ]
    bot_classes = [ HumanLikeBot ] * 5
    # observation_history = [
    #     # Round 1
    #     ('propose', (1, 2)),
    #     ('vote', (True, True, False, False, False)),
    #     ('propose', (0, 2)),
    #     ('vote', (False, True, True, True, False)),
    #     ('run', 1),
    #     # Round 2
    #     ('propose', (2, 3, 4)),
    #     ('vote', (False, True, True, True, False)),
    #     ('run', 0),
    #     # Round 3
    #     ('propose', (2, 3)),
    #     ('vote', (False, True, True, True, False)),
    #     ('run', 0),
    #     # Round 4
    #     ('propose', (2, 3, 4)),
    #     ('vote', (False, True, True, True, False)),
    #     ('run', 1),
    #     # Round 5
    #     ('propose', (2, 3, 4)),
    #     ('vote', (False, True, True, False, False)),
    #     ('propose', (1, 2, 3)),
    #     ('vote', (True, True, False, False, False)),
    #     ('propose', (0, 1, 3)),
    #     ('vote', (True, False, False, False, True)),
    # ]
    # observation_history = [
    #     # Round 1
    #     ('propose', (0, 4)),
    #     ('vote', (True, True, True, True, False)),
    #     ('run', 1),
    #     # Round 2
    #     ('propose', (1, 3, 4)),
    #     ('vote', (True, True, False, True, False)),
    #     ('run', 1),
    #     # Round 3
    #     ('propose', (1, 2)),
    #     ('vote', (True, True, False, False, False)),
    #     ('propose', (2, 3)),
    #     ('vote', (False, False, True, True, True)),
    #     ('run', 0),
    #     # Round 4
    #     ('propose', (2, 3, 4)),
    #     ('vote', (False, False, True, True, True)),
    #     ('run', 0),
    #     # # Round 5
    #     ('propose', (0, 1, 3)),
    #     ('vote', (True, True, False, False, False)),
    #     ('propose', (1, 3, 4)),
    #     ('vote', (True, True, False, False, False)),
    #     ('propose', (0, 2, 3)),
    #     ('vote', (True, True, False, False, False)),
    #     ('propose', (2, 3, 4)),
    #     ('vote', (False, False, False, True, True)),
    #     ('propose', (2, 3, 4)),
    #     ('vote', (True, True, True, False, True))
    # ]
    observation_history = [
        # Round 1
        ('propose', (0, 4)),
        ('vote', (True, True, True, True, False)),
        ('run', 1),
        # Round 2
        ('propose', (1, 3, 4)),
        ('vote', (True, True, False, True, False)),
        ('run', 1),
        # Round 3
        ('propose', (1, 2)),
        ('vote', (True, True, False, False, False)),
        ('propose', (2, 3)),
        ('vote', (False, False, True, False, True)),
        ('propose', (2, 4)),
        ('vote', (False, False, True, True, True)),
        ('run', 0),
        # Round 4
        ('propose', (2, 3, 4)),
        ('vote', (False, False, True, True, True)),
        ('run', 0),
        # # Round 5
        # person 1
        ('propose', (0, 1, 3)),
        ('vote', (True, True, False, False, False)),
        # person 2
        ('propose', (1, 3, 4)),
        ('vote', (True, True, False, False, False)),
        # person 3
        ('propose', (0, 2, 3)),
        ('vote', (True, True, False, False, False)),
    ]

    hidden_states, lls = calculate_subgame_ll(roles, 5, bot_classes, observation_history, tremble=1e-8)
    lls -= np.max(lls)
    
    probs = np.exp(lls)
    probs /= np.sum(probs)
    multiple = np.max(probs) / 50
    for hidden_state, prob in zip(hidden_states, probs):
        print "{: >10} {: >10} {: >10} {: >10} {: >10}: {prob}".format(*hidden_state, prob='#' * int(prob / multiple))

    print "Solving subgame"
    state = AvalonState(proposer=3, propose_count=3, succeeds=2, fails=2, status='propose', proposal=None, game_end=None, num_players=5)
    solve_subgame(hidden_states, lls, state, iterations=100000)


