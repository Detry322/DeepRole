import numpy as np

from battlefield.bots.bot import Bot
from battlefield.avalon_types import EVIL_ROLES, GOOD_ROLES, MissionAction, VoteAction, ProposeAction, PickMerlinAction
from battlefield.bots.deeprole.lookup_tables import get_deeprole_perspective, assignment_id_to_hidden_state, print_top_k_belief, print_top_k_viewpoint_belief
from battlefield.bots.deeprole.run_deeprole import run_deeprole_on_node
from battlefield.bots.cfr_bot import proposal_to_bitstring, bitstring_to_proposal

def get_start_node(proposer):
    return {
        "type": "TERMINAL_PROPOSE_NN",
        "succeeds": 0,
        "fails": 0,
        "propose_count": 0,
        "proposer": proposer,
        "new_belief": list(np.ones(60)/60.0),
        "nozero_belief": list(np.ones(60)/60.0)
    }

def assert_eq(a, b):
    assert a == b, "{} != {}".format(a, b)

def votes_to_bitstring(vote_moves):
    result = 0
    for i, vote in enumerate(vote_moves):
        if vote.up:
            result |= (1 << i)
    return result

def print_move_probs(probs, legal_actions, cutoff=0.95):
    zipped = zip(probs, legal_actions)
    zipped.sort(reverse=True)
    total = 0.0
    print "----- move probs -----"
    while total < cutoff and len(zipped) > 0:
        prob, move = zipped[0]
        zipped = zipped[1:]
        print move, prob
        total += prob

# Plays randomly, except always fails missions if bad.
class Deeprole(Bot):
    ITERATIONS = 100
    WAIT_ITERATIONS = 50
    NO_ZERO=False
    NN_FOLDER='deeprole_models'
    DEEPROLE_BINARY='deeprole'

    def __init__(self):
        self.belief = list(np.ones(60)/60.0)
        pass


    def reset(self, game, player, role, hidden_states):
        self.node = run_deeprole_on_node(
            get_start_node(game.proposer),
            self.ITERATIONS,
            self.WAIT_ITERATIONS,
            no_zero=self.NO_ZERO,
            nn_folder=self.NN_FOLDER,
            binary=self.DEEPROLE_BINARY
        )
        self.player = player
        self.perspective = get_deeprole_perspective(player, hidden_states[0])
        self.belief = list(np.ones(60)/60.0)
        self.thought_log = []
        # print self.perspective


    def handle_transition(self, old_state, new_state, observation, move=None):
        self.thought_log.append((
            old_state.succeeds, old_state.fails, old_state.propose_count, self.belief
        ))

        if old_state.status == 'merlin':
            return

        if old_state.status == 'propose':
            proposal = observation
            bitstring = proposal_to_bitstring(proposal)
            child_index = self.node['propose_options'].index(bitstring)
            self.node = self.node['children'][child_index]
        elif old_state.status == 'vote':
            child_index = votes_to_bitstring(observation)
            self.node = self.node['children'][child_index]
        elif old_state.status == 'run':
            num_fails = observation
            self.node = self.node['children'][num_fails]

        if self.node['type'] == 'TERMINAL_PROPOSE_NN':
            # print_top_k_belief(self.node['new_belief'])
            # print "Player {} perspective {}".format(self.player, self.perspective)
            # print_top_k_viewpoint_belief(self.node['new_belief'], self.player, self.perspective)
            # print self.node['new_belief']
            self.belief = self.node['new_belief']
            self.node = run_deeprole_on_node(
                self.node,
                self.ITERATIONS,
                self.WAIT_ITERATIONS,
                no_zero=self.NO_ZERO,
                nn_folder=self.NN_FOLDER,
                binary=self.DEEPROLE_BINARY
            )

        if self.node['type'].startswith("TERMINAL_") and self.node['type'] != "TERMINAL_MERLIN":
            return

        assert_eq(new_state.succeeds, self.node["succeeds"])
        assert_eq(new_state.fails, self.node["fails"])

        if new_state.status == 'propose':
            assert_eq(new_state.proposer, self.node['proposer'])
            assert_eq(new_state.propose_count, self.node['propose_count'])
        elif new_state.status == 'vote':
            assert_eq(new_state.proposer, self.node['proposer'])
            assert_eq(new_state.propose_count, self.node['propose_count'])
            assert_eq(new_state.proposal, bitstring_to_proposal(self.node['proposal']))
        elif new_state.status == 'run':
            assert_eq(new_state.proposal, bitstring_to_proposal(self.node['proposal']))


    def get_action(self, state, legal_actions):
        probs = self.get_move_probabilities(state, legal_actions)
        index = np.random.choice(len(legal_actions), p=probs)
        return legal_actions[index]


    def get_move_probabilities(self, state, legal_actions):
        if len(legal_actions) == 1:
            return np.array([1.0])

        if state.status == 'propose':
            probs = np.zeros(len(legal_actions))
            propose_strategy = self.node['propose_strat'][self.perspective]
            propose_options = self.node['propose_options']
            for strategy_prob, proposal_bitstring in zip(propose_strategy, propose_options):
                action = ProposeAction(proposal=bitstring_to_proposal(proposal_bitstring))
                probs[legal_actions.index(action)] = strategy_prob
            # print probs
            # print_move_probs(probs, legal_actions)
            return probs
        elif state.status == 'vote':
            probs = np.zeros(len(legal_actions))
            vote_strategy = self.node['vote_strat'][self.player][self.perspective]
            for strategy_prob, vote_up in zip(vote_strategy, [False, True]):
                action = VoteAction(up=vote_up)
                probs[legal_actions.index(action)] = strategy_prob
            return probs
        elif state.status == 'run':
            probs = np.zeros(len(legal_actions))
            mission_strategy = self.node['mission_strat'][self.player][self.perspective]
            for strategy_prob, fail in zip(mission_strategy, [False, True]):
                action = MissionAction(fail=fail)
                probs[legal_actions.index(action)] = strategy_prob
            return probs
        elif state.status == 'merlin':
            # print np.array(self.node['merlin_strat'][self.player][self.perspective])
            return np.array(self.node['merlin_strat'][self.player][self.perspective])


# For examining the effect of more iterations
class Deeprole_3_1(Deeprole):
    ITERATIONS = 3
    WAIT_ITERATIONS = 1

class Deeprole_10_5(Deeprole):
    ITERATIONS = 10
    WAIT_ITERATIONS = 5

class Deeprole_30_15(Deeprole):
    ITERATIONS = 30
    WAIT_ITERATIONS = 15

class Deeprole_100_50(Deeprole):
    ITERATIONS = 100
    WAIT_ITERATIONS = 50

class Deeprole_300_150(Deeprole):
    ITERATIONS = 300
    WAIT_ITERATIONS = 150


# For examining the effect of "Wait iterations"
class Deeprole_100_0(Deeprole):
    ITERATIONS = 100
    WAIT_ITERATIONS = 0

class Deeprole_100_25(Deeprole):
    ITERATIONS = 100
    WAIT_ITERATIONS = 25

# class Deeprole_100_50(Deeprole):

class Deeprole_100_75(Deeprole):
    ITERATIONS = 100
    WAIT_ITERATIONS = 75


class Deeprole_100_100(Deeprole):
    ITERATIONS = 100
    WAIT_ITERATIONS = 99


class Deeprole_NoZeroing(Deeprole):
    NO_ZERO = True


class Deeprole_OldNNs(Deeprole):
    NN_FOLDER = 'deeprole_models_old'



class Deeprole_NoZeroUnconstrained(Deeprole):
    DEEPROLE_BINARY = 'deeprole_nozero'
    NN_FOLDER = 'deeprole_nozero_unconstrained'


class Deeprole_ZeroingUnconstrained(Deeprole):
    DEEPROLE_BINARY = 'deeprole_zeroing'
    NN_FOLDER = 'deeprole_zeroing_unconstrained'


class Deeprole_ZeroingWinProbs(Deeprole):
    DEEPROLE_BINARY = 'deeprole_zeroing'
    NN_FOLDER = 'deeprole_zeroing_winprobs'
