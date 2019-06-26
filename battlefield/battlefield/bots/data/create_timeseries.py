import os
import gzip
import cPickle as pickle
import json
import sys
import numpy as np
from keras.preprocessing.sequence import pad_sequences

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
sys.path.append(PARENT_DIR)

from battlefield.avalon_types import filter_hidden_states, EVIL_ROLES, GOOD_ROLES, ProposeAction, VoteAction, MissionAction, PickMerlinAction, possible_hidden_states, starting_hidden_states

DATAFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'relabeled.json'))
VOTE_OUTPUT_FILENAME = os.path.abspath(os.path.join(os.path.dirname(__file__), 'vote_data.npz'))
PROPOSE_OUTPUT_FILENAME = os.path.abspath(os.path.join(os.path.dirname(__file__), 'propose_data.npz'))

print "loading input json"
with open(DATAFILE, 'r') as f:
    input_json = json.load(f)

ROLES = ['servant', 'merlin', 'percival', 'minion', 'assassin', 'mordred', 'morgana', 'oberon']

ROLE_TO_ONEHOT = {
    role: np.eye(len(ROLES))[i] for i, role in enumerate(ROLES)
}

def perspective_from_hidden_states(hidden_states):
    roles = [set([]) for _ in range(len(hidden_states[0]))]
    for hidden_state in hidden_states:
        for p, role in enumerate(hidden_state):
            roles[p].add(role)
    return tuple([ frozenset(possible) for possible in roles ])


def create_perception(hidden_states):
    perceptions = [ np.zeros(len(ROLES)) for _ in ROLES ]
    for hidden_state in hidden_states:
        for p, role in enumerate(hidden_state):
            perceptions[p] += ROLE_TO_ONEHOT[role]
    return np.array(perceptions) / len(hidden_states)


def reconstruct_hidden_state(game):
    roles = []
    for player in game['players']:
        roles.append('minion' if player['spy'] else 'servant')

    for role, p in game['roles'].items():
        if role not in set(['assassin', 'merlin', 'mordred', 'percival', 'morgana', 'oberon']):
            continue
        roles[p] = role
    return tuple(roles)


VOTE_INPUTS = [] # a list of lists
VOTE_OUTPUTS = [] # a list of 0/1s

PROPOSE_INPUTS = [] # a list of lists
PROPOSE_OUTPUTS = [] # a list of outputs


def onehot(player, num_players=5):
    res = np.zeros(num_players)
    res[player] = 1.
    return res


def process_game(game):
    try:
        hidden_state = reconstruct_hidden_state(game)
        if len(hidden_state) != 5:
            return
        print game['id']
        possible = possible_hidden_states(set(hidden_state), num_players=len(hidden_state))
        perspectives = [
            starting_hidden_states(player, hidden_state, possible)
            for player, _ in enumerate(hidden_state)
        ]
        perceptions = [
            create_perception(perspective)
            for perspective in perspectives
        ]

        vote_inputs = [ [] for _ in hidden_state ]
        propose_inputs = [ [] for _ in hidden_state ]

        for round_ in game['log']:
            last_proposal_dict = None
            last_proposal = np.zeros(5)
            last_votes = np.zeros(5)
            for proposal_num in ['1', '2', '3', '4', '5']:
                if proposal_num not in round_:
                    break
                proposal_dict = last_proposal_dict = round_[proposal_num]
                proposer = proposal_dict['proposer']

                for player, perception in enumerate(perceptions):
                    propose_inputs[player].append(np.concatenate([
                        perception.flat,
                        onehot(proposer),
                        last_proposal,
                        last_votes
                    ]))

                proposal = onehot(proposal_dict['team'])
                PROPOSE_INPUTS.append(propose_inputs[proposer][:])
                PROPOSE_OUTPUTS.append(proposal)

                votes = np.array([ -1.0 if vote == 'Reject' else 1.0 for vote in proposal_dict['votes'] ])
                for player, vote in enumerate(proposal_dict['votes']):
                    vote_inputs[player].append(np.concatenate([
                        perceptions[player].flat,
                        onehot(player),
                        proposal,
                        np.zeros(5)
                    ]))

                    VOTE_INPUTS.append(vote_inputs[player][:])
                    VOTE_OUTPUTS.append(1.0 if vote == 'Approve' else 0.0)

                    vote_inputs[player][-1] = np.concatenate([
                        perceptions[player].flat,
                        onehot(player),
                        proposal,
                        votes
                    ])

                last_proposal = proposal
                last_votes = votes


            num_fails = len([ mission_vote for mission_vote in round_['mission'] if mission_vote == 'Fail' ])
            for player, perspective in enumerate(perspectives):
                perspectives[player] = filter_hidden_states(perspectives[player], set(last_proposal_dict['team']), num_fails)
                perceptions[player] = create_perception(perspectives[player])

    except AssertionError:
        print game['id'], 'is bad'


for game in input_json:
    process_game(game)

print "Padding"
VOTE_INPUTS = pad_sequences(VOTE_INPUTS)
PROPOSE_INPUTS = pad_sequences(PROPOSE_INPUTS)

print "saving"
np.savez_compressed(VOTE_OUTPUT_FILENAME, VOTE_INPUTS, np.array(VOTE_OUTPUTS))
np.savez_compressed(PROPOSE_OUTPUT_FILENAME, PROPOSE_INPUTS, np.array(PROPOSE_OUTPUTS))
