import random
import numpy as np

from battlefield.bots.bot import Bot
from battlefield.avalon_types import VoteAction

# Plays randomly
class RandomBot(Bot):
    def __init__(self):
        pass

    def reset(self, game, player, role, hidden_states):
        self.game = game
        self.player = player
        self.role = role
        self.hidden_states = hidden_states


    def handle_transition(self, old_state, new_state, observation, move=None):
        pass


    def get_action(self, state, legal_actions):
        return random.choice(legal_actions)


    def get_move_probabilities(self, state, legal_actions):
        return np.ones(len(legal_actions)) / len(legal_actions)


class RandomBotUV(Bot):
    def __init__(self):
        pass


    def reset(self, game, player, role, hidden_states):
        self.game = game
        self.player = player
        self.role = role
        self.hidden_states = hidden_states


    def handle_transition(self, old_state, new_state, observation, move=None):
        pass


    def get_action(self, state, legal_actions):
        if state.status == 'vote':
            return VoteAction(up=True)

        return random.choice(legal_actions)


    def get_move_probabilities(self, state, legal_actions):
        if state.status == 'vote':
            result = np.zeros(len(legal_actions))
            result[legal_actions.index(VoteAction(up=True))] = 1.
            return result

        return np.ones(len(legal_actions)) / len(legal_actions) 
