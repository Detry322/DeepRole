import itertools

NUM_PLAYERS = 5

print """\
#include "lookup_tables.h"

const int PROPOSAL_TO_INDEX_LOOKUP[32] = {-1, -1, -1, 0, -1, 1, 4, 0, -1, 2, 5, 1, 7, 3, 6, -1, -1, 3, 6, 2, 8, 4, 7, -1, 9, 5, 8, -1, 9, -1, -1, -1};
const int INDEX_TO_PROPOSAL_2[10] = {3, 5, 9, 17, 6, 10, 18, 12, 20, 24};
const int INDEX_TO_PROPOSAL_3[10] = {7, 11, 19, 13, 21, 25, 14, 22, 26, 28};
const int ROUND_TO_PROPOSE_SIZE[5] = {2, 3, 2, 3, 3};
"""

VIEWPOINT_TO_BAD = [[] for _ in range(NUM_PLAYERS)]

for player, viewpoint_arr in enumerate(VIEWPOINT_TO_BAD):
    viewpoint_arr.append(-1) # first viewpoint is neutral
    
    remaining_players = [p for p in range(NUM_PLAYERS) if p != player]
    for bad_guys in itertools.combinations(remaining_players, 2):
        n = (1 << bad_guys[0]) | (1 << bad_guys[1])
        viewpoint_arr.append(n)

    viewpoint_arr.extend(remaining_players) # Partner in crime if assassin
    viewpoint_arr.extend(remaining_players) # Partner in crime if minion

print """
const int VIEWPOINT_TO_BAD[NUM_PLAYERS][NUM_VIEWPOINTS] = {
    %s
};
""" % (",\n    ".join([
    ("{" + ",  ".join(["{:> 3}".format(v) for v in viewpoint_arr]) + " }")
    for viewpoint_arr in VIEWPOINT_TO_BAD
]))


ASSIGNMENT_TO_VIEWPOINT = []
ASSIGNMENT_TO_EVIL = []
ASSIGNMENT_TO_ROLES = []

for assignment in itertools.permutations(range(NUM_PLAYERS), 3):
    merlin, assassin, minion = assignment

    ASSIGNMENT_TO_ROLES.append(assignment)

    viewpoint = [0] * NUM_PLAYERS

    bad = (1 << assassin) | (1 << minion)
    ASSIGNMENT_TO_EVIL.append(bad)
    merlin_viewpoint = 1 + VIEWPOINT_TO_BAD[merlin][1:7].index(bad)
    assassin_viewpoint = 7 + VIEWPOINT_TO_BAD[assassin][7:11].index(minion)
    minion_viewpoint = 11 + VIEWPOINT_TO_BAD[minion][11:15].index(assassin)

    viewpoint[merlin] = merlin_viewpoint
    viewpoint[assassin] = assassin_viewpoint
    viewpoint[minion] = minion_viewpoint
    ASSIGNMENT_TO_VIEWPOINT.append(viewpoint)

print """
const int ASSIGNMENT_TO_VIEWPOINT[NUM_ASSIGNMENTS][NUM_PLAYERS] = {
    %s
};
""" % (",\n    ".join([
    ("{" + ",  ".join(["{:> 3}".format(v) for v in viewpoint_arr]) + " }")
    for viewpoint_arr in ASSIGNMENT_TO_VIEWPOINT
]))

print """
const int ASSIGNMENT_TO_EVIL[NUM_ASSIGNMENTS] = %s;
""" % ("{" + ", ".join(["{:> 3}".format(v) for v in ASSIGNMENT_TO_EVIL]) + " }")

print """
const int ASSIGNMENT_TO_ROLES[NUM_ASSIGNMENTS][3] = {
    %s
};
""" % (",\n    ".join([
    ("{" + ",  ".join(["{:> 3}".format(v) for v in assignment_arr]) + " }")
    for assignment_arr in ASSIGNMENT_TO_ROLES
]))

VIEWPOINT_TO_PARTNER_VIEWPOINT = [[] for _ in range(NUM_PLAYERS)]

for player, viewpoint_to_bad in enumerate(VIEWPOINT_TO_BAD):
    for viewpoint_index, partner in enumerate(viewpoint_to_bad):
        if viewpoint_index < 7:
            VIEWPOINT_TO_PARTNER_VIEWPOINT[player].append(-1)
            continue

        is_assassin = (viewpoint_index < 11)

        if is_assassin:
            partner_viewpoint = 11 + VIEWPOINT_TO_BAD[partner][11:15].index(player)
        else:
            partner_viewpoint = 7 + VIEWPOINT_TO_BAD[partner][7:11].index(player)

        VIEWPOINT_TO_PARTNER_VIEWPOINT[player].append(partner_viewpoint)

print """
const int VIEWPOINT_TO_PARTNER_VIEWPOINT[NUM_PLAYERS][NUM_VIEWPOINTS] = {
    %s
};
""" % (",\n    ".join([
    ("{" + ",  ".join(["{:> 3}".format(v) for v in viewpoint_arr]) + " }")
    for viewpoint_arr in VIEWPOINT_TO_PARTNER_VIEWPOINT
]))
