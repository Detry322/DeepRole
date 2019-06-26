import random
import numpy as np

from battlefield.bots.bot import Bot
from battlefield.avalon_types import EVIL_ROLES, GOOD_ROLES, MissionAction, VoteAction, PickMerlinAction

# Plays randomly, except always fails missions if bad.
class SimpleStatsBot(Bot):
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
        probs = self.get_move_probabilities(state, legal_actions)
        index = np.random.choice(len(legal_actions), p=probs)
        return legal_actions[index]


    def get_move_probabilities(self, state, legal_actions):
        result = np.zeros(len(legal_actions))
        if state.status == 'propose':
            # Only consider proposals with yourself
            for i, act in enumerate(legal_actions):
                result[i] += 1 if self.player in act.proposal else 0
        elif state.status == 'vote':
            result += 1
            if self.player in state.proposal:
                # Vote up most proposals with yourself
                result[legal_actions.index(VoteAction(up=True))] += 5
            elif self.player not in state.proposal:
                if state.propose_count == 4 and not self.is_evil:
                    # Vote up most proposals on the final round if you're good
                    result[legal_actions.index(VoteAction(up=True))] += 5
                else:
                    # Vote down most proposals which don't contain you.
                    result[legal_actions.index(VoteAction(up=False))] += 5
        elif state.status == 'run':
            result += 1
            if self.is_evil:
                # Fail most missions unless it's the first one
                if state.fails + state.succeeds == 0:
                    result[legal_actions.index(MissionAction(fail=False))] += 5
                else:
                    result[legal_actions.index(MissionAction(fail=True))] += 5
        elif state.status == 'merlin':
            # Try to pick merlin based off of hidden states
            for hidden_state in self.hidden_states:
                merlin = hidden_state.index('merlin')
                result[legal_actions.index(PickMerlinAction(merlin=merlin))] += 1

        return result / np.sum(result)


