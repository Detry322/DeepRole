import itertools
import numpy as np

hidden_state_to_assignment_id = {}

assignment_id_to_hidden_state = []

for i, (merlin, assassin, minion) in enumerate(itertools.permutations(range(5), 3)):
    hidden_state = ['servant']*5
    hidden_state[merlin] = 'merlin'
    hidden_state[assassin] = 'assassin'
    hidden_state[minion] = 'minion'
    hidden_state_to_assignment_id[tuple(hidden_state)] = i
    assignment_id_to_hidden_state.append(tuple(hidden_state))


def get_deeprole_perspective(player, hidden_state):
    assignemnt_id = hidden_state_to_assignment_id[tuple(hidden_state)]
    return ASSIGNMENT_TO_VIEWPOINT[assignemnt_id][player]


def print_top_k_belief(belief, k=5):
    stuff = zip(belief, assignment_id_to_hidden_state)
    stuff.sort(reverse=True)
    print "===== Third person belief ===="
    for v, h in stuff[:k]:
        print "{: >45}: {}".format(h, v)


def print_top_k_viewpoint_belief(belief, player, viewpoint, k=5):
    valid_beliefs = []
    valid_hidden_states = []
    for i, viewpoints in enumerate(ASSIGNMENT_TO_VIEWPOINT):
        if viewpoints[player] == viewpoint:
            valid_beliefs.append(belief[i])
            valid_hidden_states.append(assignment_id_to_hidden_state[i])

    s = np.sum(valid_beliefs)

    zipped = zip(valid_beliefs, valid_hidden_states)
    zipped.sort(reverse=True)

    print "===== Actual belief for player {} ====".format(player)
    for value, assignment in zipped[:k]:
        print "{: >45}: {: >10} {: >10}".format(assignment, value, value/s)


ASSIGNMENT_TO_VIEWPOINT = [
    [  1,    8,   12,    0,    0 ],
    [  2,    9,    0,   12,    0 ],
    [  3,   10,    0,    0,   12 ],
    [  1,   12,    8,    0,    0 ],
    [  4,    0,    9,   13,    0 ],
    [  5,    0,   10,    0,   13 ],
    [  2,   13,    0,    8,    0 ],
    [  4,    0,   13,    9,    0 ],
    [  6,    0,    0,   10,   14 ],
    [  3,   14,    0,    0,    8 ],
    [  5,    0,   14,    0,    9 ],
    [  6,    0,    0,   14,   10 ],
    [  8,    1,   11,    0,    0 ],
    [  9,    2,    0,   11,    0 ],
    [ 10,    3,    0,    0,   11 ],
    [ 12,    1,    7,    0,    0 ],
    [  0,    4,    9,   13,    0 ],
    [  0,    5,   10,    0,   13 ],
    [ 13,    2,    0,    7,    0 ],
    [  0,    4,   13,    9,    0 ],
    [  0,    6,    0,   10,   14 ],
    [ 14,    3,    0,    0,    7 ],
    [  0,    5,   14,    0,    9 ],
    [  0,    6,    0,   14,   10 ],
    [  7,   11,    1,    0,    0 ],
    [  9,    0,    2,   11,    0 ],
    [ 10,    0,    3,    0,   11 ],
    [ 11,    7,    1,    0,    0 ],
    [  0,    9,    4,   12,    0 ],
    [  0,   10,    5,    0,   12 ],
    [ 13,    0,    2,    7,    0 ],
    [  0,   13,    4,    8,    0 ],
    [  0,    0,    6,   10,   14 ],
    [ 14,    0,    3,    0,    7 ],
    [  0,   14,    5,    0,    8 ],
    [  0,    0,    6,   14,   10 ],
    [  7,   11,    0,    1,    0 ],
    [  8,    0,   11,    2,    0 ],
    [ 10,    0,    0,    3,   11 ],
    [ 11,    7,    0,    1,    0 ],
    [  0,    8,   12,    4,    0 ],
    [  0,   10,    0,    5,   12 ],
    [ 12,    0,    7,    2,    0 ],
    [  0,   12,    8,    4,    0 ],
    [  0,    0,   10,    6,   13 ],
    [ 14,    0,    0,    3,    7 ],
    [  0,   14,    0,    5,    8 ],
    [  0,    0,   14,    6,    9 ],
    [  7,   11,    0,    0,    1 ],
    [  8,    0,   11,    0,    2 ],
    [  9,    0,    0,   11,    3 ],
    [ 11,    7,    0,    0,    1 ],
    [  0,    8,   12,    0,    4 ],
    [  0,    9,    0,   12,    5 ],
    [ 12,    0,    7,    0,    2 ],
    [  0,   12,    8,    0,    4 ],
    [  0,    0,    9,   13,    6 ],
    [ 13,    0,    0,    7,    3 ],
    [  0,   13,    0,    8,    5 ],
    [  0,    0,   13,    9,    6 ]
]
