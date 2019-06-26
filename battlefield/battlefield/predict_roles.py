import pandas as pd
import numpy as np
import copy
import gzip
import itertools
import multiprocessing
import sys

from battlefield.compare_to_human import reconstruct_hidden_state, load_human_data
from battlefield.avalon_types import GOOD_ROLES, EVIL_ROLES, AVALON_PLAYER_COUNT, possible_hidden_states, starting_hidden_states, MissionAction, VoteAction, ProposeAction
from battlefield.avalon import AvalonState

def human_game_state_generator(avalon_start, human_game, hidden_state):
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
            yield state, new_state, observation
            state = new_state

            assert state.status == 'vote'
            moves = [VoteAction(up=(vote == 'Approve')) for vote in proposal['votes']]
            new_state, _, observation = state.transition(moves, hidden_state)
            yield state, new_state, observation
            state = new_state

            if state.status == 'run':
                break

        secret_votes = sorted(zip(last_proposal['team'], round_['mission']))
        moves = [MissionAction(fail=(vote == "Fail")) for player, vote in secret_votes]
        new_state, _, observation = state.transition(moves, hidden_state)
        yield state, new_state, observation
        state = new_state


__hidden_state_to_perspectives = {}
def get_player_perspectives_for_hidden_state(all_hidden, hidden_state):
    global __hidden_state_to_perspectives
    if hidden_state not in __hidden_state_to_perspectives:
        __hidden_state_to_perspectives[hidden_state] = [
            starting_hidden_states(player, hidden_state, all_hidden)
            for player, _ in enumerate(hidden_state)
        ]
    return __hidden_state_to_perspectives[hidden_state]


__all_possible_perspectives = {}
def get_possible_perspectives(hidden_state):
    global __all_possible_perspectives
    roles = (frozenset(hidden_state), len(hidden_state))
    if roles not in __all_possible_perspectives:
        all_possible_hidden = possible_hidden_states(*roles)
        __all_possible_perspectives[roles] = (all_possible_hidden, [
            get_player_perspectives_for_hidden_state(all_possible_hidden, hidden)
            for hidden in all_possible_hidden
        ])
    return __all_possible_perspectives[roles]


def handle_propose_particle_transition(old_state, new_state, proposal, particle, tremble):
    assert old_state.status == 'propose'
    proposal = ProposeAction(proposal)
    proposer = old_state.proposer
    legal_actions = old_state.legal_actions(proposer, particle['hidden_state'])
    action_index = legal_actions.index(proposal)
    move_probs = particle['bots'][proposer].get_move_probabilities(old_state, legal_actions)
    particle['prob'] *= ( tremble * 1.0/len(legal_actions) + (1.0 - tremble) * move_probs[action_index] )
    
    for player, bot in enumerate(particle['bots']):
        bot.handle_transition(old_state, new_state, proposal, move=proposal if player == proposer else None)

    return [particle] if particle['prob'] != 0 else []


def handle_vote_particle_transition(old_state, new_state, votes, particle, tremble):
    assert old_state.status == 'vote'

    legal_actions = old_state.legal_actions(0, particle['hidden_state'])

    for player, bot in enumerate(particle['bots']):
        action = votes[player]
        move_probs = bot.get_move_probabilities(old_state, legal_actions)
        particle['prob'] *= ( tremble * 1.0 / len(legal_actions) + (1.0 - tremble) * move_probs[legal_actions.index(action)] )
        bot.handle_transition(old_state, new_state, votes, move=action)

    return [particle] if particle['prob'] != 0 else []


def handle_mission_particle_transition(old_state, new_state, num_fails, particle, tremble):
    assert old_state.status == 'run'
    bad_on_mission = [ p for p in old_state.proposal if particle['hidden_state'][p] in EVIL_ROLES ]
    if num_fails > len(bad_on_mission):
        return []

    new_particles = []
    for failers in itertools.combinations(bad_on_mission, r=num_fails):
        new_particle = copy.deepcopy(particle)
        new_particle['disambiguation'].append((old_state, failers))
        new_particles.append(new_particle)

        for player, bot in enumerate(new_particle['bots']):
            if player not in old_state.proposal:
                bot.handle_transition(old_state, new_state, num_fails, move=None)
                continue

            if player not in bad_on_mission:
                bot.handle_transition(old_state, new_state, num_fails, move=MissionAction(fail=False))
                continue

            move = MissionAction(fail=True) if player in failers else MissionAction(fail=False)
            legal_actions = old_state.legal_actions(player, particle['hidden_state'])
            move_probs = bot.get_move_probabilities(old_state, legal_actions)

            new_particle['prob'] *= ( tremble * 1.0 / len(legal_actions) + (1.0 - tremble) * move_probs[legal_actions.index(move)])
            bot.handle_transition(old_state, new_state, num_fails, move=move)

    return [p for p in new_particles if p['prob'] != 0]




def handle_particle_transition(old_state, new_state, observation, particle, tremble):
    if old_state.status == 'propose':
        return handle_propose_particle_transition(old_state, new_state, observation, particle, tremble)
    elif old_state.status == 'vote':
        return handle_vote_particle_transition(old_state, new_state, observation, particle, tremble)
    elif old_state.status == 'run':
        return handle_mission_particle_transition(old_state, new_state, observation, particle, tremble)
    
    assert False, "Something went wrong"


def get_hidden_state_nll_for_game(avalon_start, game_generator, real_hidden_state, bot_class, tremble):
    all_possible_hidden, all_player_perspectives = get_possible_perspectives(real_hidden_state)
    particles = [
        {
            'prob': 1.0 / len(all_possible_hidden),
            'bots': [
                bot_class.create_and_reset(avalon_start, player, hidden_state[player], perspective)
                for player, perspective in enumerate(player_perspectives)
            ],
            'hidden_state': hidden_state,
            'disambiguation': []
        }
        for hidden_state, player_perspectives in zip(all_possible_hidden, all_player_perspectives)
    ]

    for old_state, new_state, observation in game_generator:
        probability_total = 0.0
        new_particles = []
        for particle in particles:
            transitioned_particles = handle_particle_transition(old_state, new_state, observation, particle, tremble)
            probability_total += sum([ p['prob'] for p in transitioned_particles ])
            new_particles.extend(transitioned_particles)
        particles = new_particles
        for p in particles:
            p['prob'] /= probability_total

    prob_by_hidden_state = {}
    for particle in particles:
        prob_by_hidden_state[particle['hidden_state']] = prob_by_hidden_state.get(particle['hidden_state'], 0) + particle['prob']

    filtered_particles = sorted([ { 'hidden_state': k, 'prob': v } for k, v in prob_by_hidden_state.items() if v != 0 ], key=lambda a: a['prob'], reverse=True)

    if prob_by_hidden_state.get(real_hidden_state, 0) == 0:
        print "Something went wrong..."
        return float('inf'), filtered_particles

    return -np.log(prob_by_hidden_state[real_hidden_state]), filtered_particles


def predict_evil_over_human_game(game, as_bot, tremble):
    try:
        print game['id']
        sys.stdout.flush()
        hidden_state = reconstruct_hidden_state(game)
        assert frozenset(hidden_state) == frozenset(['merlin', 'assassin', 'minion', 'servant'])
        avalon_start = AvalonState.start_state(len(hidden_state))
        game_generator = human_game_state_generator(avalon_start, game, hidden_state)
        nll, particles = get_hidden_state_nll_for_game(avalon_start, game_generator, hidden_state, as_bot, tremble)
        data = {
            'bot': as_bot.__name__,
            'game': game['id'],
            'num_players': len(hidden_state),
            'trembling_hand_prob': tremble,
            'nll': nll
        }
        for p in range(len(hidden_state)):
            data['player_{}'.format(p)] = game['players'][p]['player_id']
            data['role_{}'.format(p)] = hidden_state[p]

        return data, particles
    except AssertionError:
        return None, None


def wrap_func(args):
    return predict_evil_over_human_game(*args)


def predict_evil_over_human_data(as_bot, tremble):
    print "Computing data for {} with tremble {}".format(as_bot.__name__, tremble)
    pool = multiprocessing.Pool()
    human_data = load_human_data()
    arguments = [
        (game, as_bot, tremble) for game in human_data
        if len(game['players']) == 5 and 'merlin' in game['roles'] and 'assassin' in game['roles']
    ]
    results = pool.map(wrap_func, arguments)
    dataframe_data = []
    all_particles = {}
    for data, particles in results:
        if data is None:
            continue
        dataframe_data.append(data)
        all_particles[data['game']] = particles

    return pd.DataFrame(dataframe_data), all_particles


import itertools
def teams_iterator(num_players, who_failed):
    _, num_evil = AVALON_PLAYER_COUNT[num_players]
    for evil in itertools.combinations(range(num_players), r=num_evil):
        if all(sum(evil[p] for p in mission) >= observation for mission, observation in who_failed):
            yield tuple([i in evil for i in range(num_players)])


def most_likely_team(team_probabilities, num_players, who_failed, actual):
    lls = []
    assignments = list(teams_iterator(num_players, who_failed))
    for assignment in assignments:
        ll = 0.0
        for i in range(len(assignment)):
            for j in range(i+1, len(assignment)):
                same_team = assignment[i] == assignment[j]
                ll += np.log(team_probabilities[i][j] if same_team else (1.0 - team_probabilities[i][j]))
        lls.append(ll)
    return max(zip(lls, assignments)), lls[assignments.index(actual)]


def update_pairings(votes, vote_same_if_same_prob, vote_same_if_diff_prob, team_probabilities):
    for i, vote_i in enumerate(votes):
        for j, vote_j in enumerate(votes):
            pr_o_same_team = vote_same_if_same_prob if vote_i == vote_j else (1.0 - vote_same_if_same_prob)
            pr_o_diff_team = vote_same_if_diff_prob if vote_i == vote_j else (1.0 - vote_same_if_diff_prob)
            pr_o = team_probabilities[i][j] * pr_o_same_team + (1.0 - team_probabilities[i][j]) * pr_o_diff_team
            team_probabilities[i][j] = pr_o_same_team * team_probabilities[i][j] / pr_o



def predict_evil_using_voting_on_game(game, vote_same_if_same_prob, vote_same_if_diff_prob):
    try:
        dataframe_data = []
        hidden_state = reconstruct_hidden_state(game)
        assert len(hidden_state) <= 7
        team = tuple([role in EVIL_ROLES for role in hidden_state])
        avalon_start = AvalonState.start_state(len(hidden_state))
        game_generator = human_game_state_generator(avalon_start, game, hidden_state)

        same_team_prob = np.ones((len(hidden_state), len(hidden_state))) / 2

        vote_count = 0
        mission_num = 0
        who_failed = []
        for old_state, new_state, observation in game_generator:
            if old_state.status == 'run':
                mission_num += 1
                if old_state.fails > new_state.fails:
                    who_failed.append((old_state.proposal, observation))
            if old_state.status == 'vote':
                vote_count += 1
                update_pairings(observation, vote_same_if_same_prob, vote_same_if_diff_prob, same_team_prob)
                (nll_picked, pick), nll_correct = most_likely_team(same_team_prob, len(hidden_state), who_failed, team)
                dataframe_data.append({
                    'game': game['id'],
                    'num_players': len(hidden_state),
                    'vsis_prob': vote_same_if_same_prob,
                    'vsid_prob': vote_same_if_diff_prob,
                    'mission': mission_num,
                    'vote_count': vote_count,
                    'nll_correct': nll_correct,
                    'nll_picked': nll_picked,
                    'correct': pick == team,
                })
        sys.stdout.flush()
        return dataframe_data
    except ValueError:
        return []
    except AssertionError:
        return []


def wrapper(args):
    return predict_evil_using_voting_on_game(*args)


def predict_evil_using_voting(pool, vote_same_if_same_prob=0.7, vote_same_if_diff_prob=0.3):
    print "Starting {} {}".format(vote_same_if_same_prob, vote_same_if_diff_prob)
    sys.stdout.flush()
    human_data = load_human_data()

    results = pool.map(wrapper, [(game, vote_same_if_same_prob, vote_same_if_diff_prob) for game in human_data])

    dataframe_data = []
    map(dataframe_data.extend, results)

    df = pd.DataFrame(dataframe_data)

    print "Writing dataframe for {} {}".format(vote_same_if_same_prob, vote_same_if_diff_prob)
    sys.stdout.flush()
    with gzip.open("voting_predict/{}_{}.msg.gz".format(vote_same_if_same_prob, vote_same_if_diff_prob), 'w') as f:
        df.to_msgpack(f)


def grid_search():
    pool = multiprocessing.Pool(40)
    for a in np.arange(0.1, 0.91, 0.02):
        for b in np.arange(0.1, 0.91, 0.02):
            predict_evil_using_voting(pool, a, b)

