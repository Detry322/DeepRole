import random
import numpy as np

from battlefield.bots.bot import Bot
from battlefield.avalon_types import EVIL_ROLES, GOOD_ROLES, MissionAction, VoteAction

# Plays randomly, except always fails missions if bad.
class SimpleBot(Bot):
    def __init__(self):
        pass

    def reset(self, game, player, role, hidden_states):
        self.game = game
        self.player = player
        self.role = role
        self.hidden_states = hidden_states
        self.is_evil = role in EVIL_ROLES



    def handle_transition(self, old_state, new_state, observation, move=None):
        pass


    def get_action(self, state, legal_actions):
        if state.status == 'run':
            return MissionAction(fail=self.is_evil)
        if state.status == 'vote':
            return VoteAction(up=True)

        return random.choice(legal_actions)


    def get_move_probabilities(self, state, legal_actions):
        result = np.zeros(len(legal_actions))
        if state.status == 'run':
            result[legal_actions.index(MissionAction(fail=self.is_evil))] = 1
        elif state.status == 'vote':
            result[legal_actions.index(VoteAction(up=True))] = 1
        else:
            result[:] = 1

        return result / len(legal_actions)


