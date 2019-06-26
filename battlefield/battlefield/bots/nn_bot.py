import random
import numpy as np
import os

from battlefield.bots.bot import Bot
from battlefield.bots.observe_bot import ObserveBot
from battlefield.avalon_types import VoteAction, ProposeAction, MissionAction, filter_hidden_states

PROPOSE_MODELFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'propose_model.h5'))
VOTE_MODELFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data', 'vote_model.h5'))

PROPOSE_MODEL = None
VOTE_MODEL = None

ROLES = ['servant', 'merlin', 'percival', 'minion', 'assassin', 'mordred', 'morgana', 'oberon']

ROLE_TO_ONEHOT = {
    role: np.eye(len(ROLES))[i] for i, role in enumerate(ROLES)
}

def load_models():
    global PROPOSE_MODEL
    global VOTE_MODEL
    if PROPOSE_MODEL is None:
        import keras
        PROPOSE_MODEL = keras.models.load_model(PROPOSE_MODELFILE)
        VOTE_MODEL = keras.models.load_model(VOTE_MODELFILE)


def create_perception(hidden_states):
    perceptions = [ np.zeros(len(ROLES)) for _ in ROLES ]
    for hidden_state in hidden_states:
        for p, role in enumerate(hidden_state):
            perceptions[p] += ROLE_TO_ONEHOT[role]
    return np.array(perceptions) / len(hidden_states)


def onehot(player, num_players=5):
    res = np.zeros(num_players)
    if isinstance(player, int):
        res[player] = 1.
    else:
        res[list(player)] = 1.
    return res

# Plays via two neural nets
class NNBot(Bot):
    def __init__(self):
        pass


    def reset(self, game, player, role, hidden_states):
        load_models()
        self.game = game
        self.player = player
        self.role = role
        self.hidden_states = hidden_states
        self.perception = create_perception(self.hidden_states)
        self.propose_nn_input = [
            np.concatenate([
                self.perception.flat,
                onehot(0),
                np.zeros(5),
                np.zeros(5)
            ])
        ]
        self.vote_nn_input = []


    def handle_transition(self, old_state, new_state, observation, move=None):
        if old_state.status == 'vote':
            self.propose_nn_input.append(np.concatenate([
                self.perception.flat,
                onehot(old_state.proposer),
                onehot(old_state.proposal),
                np.array([ 1.0 if vote.up else -1.0 for vote in observation ])
            ]))
            self.vote_nn_input.append(np.concatenate([
                self.perception.flat,
                onehot(self.player),
                onehot(old_state.proposal),
                np.array([ 1.0 if vote.up else -1.0 for vote in observation ])
            ]))


        if old_state.status == 'run':
            # if move is not None and self.role in EVIL_ROLES and not move.fail:
            #     observation += 1
            self.hidden_states = filter_hidden_states(self.hidden_states, old_state.proposal, observation)
            self.perception = create_perception(self.hidden_states)

            if new_state.status == 'propose':
                self.propose_nn_input.append(np.concatenate([
                    self.perception.flat,
                    onehot(new_state.proposer),
                    np.zeros(5),
                    np.zeros(5)
                ]))


    def get_action(self, state, legal_actions):
        probs = self.get_move_probabilities(state, legal_actions)
        index = np.random.choice(len(legal_actions), p=probs)
        return legal_actions[index]


    def get_move_probabilities(self, state, legal_actions):
        if state.status == 'propose':
            probs = PROPOSE_MODEL.predict(np.array([self.propose_nn_input]))[0]
            result = np.zeros(len(legal_actions))
            for i, action in enumerate(legal_actions):
                result[i] = np.exp(sum([np.log(probs[p]) for p in action.proposal]))
            return result / np.sum(result)
        elif state.status == 'vote':
            self.vote_nn_input.append(np.concatenate([
                self.perception.flat,
                onehot(self.player),
                onehot(state.proposal),
                np.zeros(5)
            ]))
            up_vote_percent = VOTE_MODEL.predict(np.array([self.vote_nn_input]))[0]
            self.vote_nn_input.pop()
            result = np.zeros(len(legal_actions))
            up_index = legal_actions.index(VoteAction(up=True))
            result[up_index] = up_vote_percent
            result[1-up_index] = 1-up_vote_percent
            return result / np.sum(result)
        else:
           return np.ones(len(legal_actions)) / len(legal_actions)


# Plays via two neural nets
class NNBotWithObservePropose(Bot):
    def __init__(self, game, player, role, hidden_states):
        self.nn_bot = NNBot(game, player, role, hidden_states)
        self.observe_bot = ObserveBot(game, player, role, hidden_states)


    def handle_transition(self, old_state, new_state, observation, move=None):
        self.observe_bot.handle_transition(old_state, new_state, observation, move)
        self.nn_bot.handle_transition(old_state, new_state, observation, move)


    def get_action(self, state, legal_actions):
        if state.status == 'propose':
            return self.observe_bot.get_action(state, legal_actions)
        return self.nn_bot.get_action(state, legal_actions)


    def get_move_probabilities(self, state, legal_actions):
        if state.status == 'propose':
            return self.observe_bot.get_move_probabilities(state, legal_actions)
        return self.nn_bot.get_move_probabilities(state, legal_actions)
