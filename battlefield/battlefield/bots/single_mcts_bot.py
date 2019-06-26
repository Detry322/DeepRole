import random
import numpy as np

from battlefield.bots.bot import Bot
from battlefield.avalon_types import EVIL_ROLES, GOOD_ROLES, MissionAction, VoteAction, PickMerlinAction, filter_hidden_states
OPPONENT_TREMBLE = 0.1


def get_opponent_moves_and_probs(state, hidden_state, player, no_tremble=False):
    legal_actions = state.legal_actions(player, hidden_state)
    if len(legal_actions) == 1:
        return legal_actions, np.ones(1)

    move_probs = np.zeros(len(legal_actions))
    if state.status == 'propose':
        for i, act in enumerate(legal_actions):
            if player in act.proposal:
                move_probs[i] += 1
    elif state.status == 'vote':
        vote_up = player in state.proposal
        move_probs[legal_actions.index(VoteAction(up=vote_up))] = 1
    elif state.status == 'run':
        # must be evil since we have two options here.
        move_probs[legal_actions.index(MissionAction(fail=True))] = 1
    elif state.status == 'merlin':
        for i, act in enumerate(legal_actions):
            if hidden_state[act.merlin] in GOOD_ROLES:
                move_probs[i] += 1 

    move_probs /= np.sum(move_probs)
    if no_tremble:
        return legal_actions, move_probs

    tremble = np.ones(len(legal_actions)) / len(legal_actions)
    return legal_actions, (1 - OPPONENT_TREMBLE) * move_probs + OPPONENT_TREMBLE * tremble


def select_opponent_move(state, hidden_state, player, no_tremble=False):
    moves, probs = get_opponent_moves_and_probs(state, hidden_state, player, no_tremble=no_tremble)
    return moves[np.random.choice(len(moves), p=probs)]


class Node:
    def __init__(self, legal_actions, incoming_edge, parent, is_terminal, terminal_value=None):
        self.parent = parent
        self.incoming_edge = incoming_edge
        self.is_terminal = is_terminal
        if self.is_terminal:
            self.terminal_value = terminal_value
        else:
            self.choose_counts = { action: 0 for action in legal_actions }
            self.total_payoffs = { action: 0.0 for action in legal_actions }
            self.legal_actions = legal_actions
            self.children = {}
            self.total_choices = 0

    def select_move(self):
        ucb_cur_max = -float('inf')
        best_move = None
        unseen_moves = []
        for move in self.legal_actions:
            choose_count = self.choose_counts[move]
            if choose_count == 0:
                unseen_moves.append(move)
                continue
            ucb_val = self.total_payoffs[move] / self.choose_counts[move] + (2*np.log(self.total_choices)/self.choose_counts[move])**0.5
            if ucb_val > ucb_cur_max:
                ucb_cur_max = ucb_val
                best_move = move
        if len(unseen_moves) != 0:
            return unseen_moves[np.random.choice(len(unseen_moves))]
        return best_move


def hidden_state_iterator(hidden_states, num_searches):
    count = 0
    while count < num_searches:
        for index in np.random.permutation(len(hidden_states)):
            count += 1
            yield hidden_states[index]
            if count >= num_searches:
                break


def next_node(node, state, hidden_state, player, move):
    moves = [ move if p == player else select_opponent_move(state, hidden_state, p) for p in state.moving_players() ]
    state, _, _  = state.transition(moves, hidden_state)
    while not state.is_terminal() and player not in state.moving_players():
        moves = [select_opponent_move(state, hidden_state, p) for p in state.moving_players()]
        state, _, _ = state.transition(moves, hidden_state)

    key = (move, state.as_key())
    if key in node.children:
        return node.children[key], state, False

    is_terminal = state.is_terminal()
    legal_actions = None if is_terminal else state.legal_actions(player, hidden_state)
    terminal_value = state.terminal_value(hidden_state)[player] if is_terminal else None
    new_node = Node(legal_actions, key, node, is_terminal, terminal_value)
    node.children[key] = new_node
    return new_node, state, True



def find_leaf_and_payoff(node, state, hidden_state, player, node_value_func):
    if node.is_terminal:
        assert state.is_terminal(), "Terminal node not terminal"
        return node, node.terminal_value

    move = node.select_move()
    new_node, next_state, is_new = next_node(node, state, hidden_state, player, move)
    if is_new:
        payoff = node_value_func(state, hidden_state, player)
        return new_node, payoff

    return find_leaf_and_payoff(new_node, next_state, hidden_state, player, node_value_func)


def search_and_backprop(node, state, hidden_state, player, node_value_func):
    node, payoff = find_leaf_and_payoff(node, state, hidden_state, player, node_value_func)
    while node.parent is not None:
        parent_action, _ = node.incoming_edge
        node.parent.total_choices += 1
        node.parent.choose_counts[parent_action] += 1
        node.parent.total_payoffs[parent_action] += payoff
        node = node.parent


def search_mcts(state, player, hidden_states, node_value_func, num_searches=100):
    root = None
    for hidden_state in hidden_state_iterator(hidden_states, num_searches):
        if root is None:
            root = Node(state.legal_actions(player, hidden_state), None, None, is_terminal=False)

        search_and_backprop(root, state, hidden_state, player, node_value_func)
    return root.select_move()


NUM_PLAYOUTS = 5
def playout_value_func(root_state, hidden_state, player):
    total_payoff = 0
    for _ in range(NUM_PLAYOUTS):
        state = root_state
        while not state.is_terminal():
            moves = [ select_opponent_move(state, hidden_state, p) for p in state.moving_players() ]
            state, _, _ = state.transition(moves, hidden_state)
        total_payoff += state.terminal_value(hidden_state)[player]
    return total_payoff


# Assumes each mission has a 50-50 chance of failing or succeeding
GOOD_PASS_THREE_PROB = {
    (0, 0): 0.5,
    (0, 1): 0.3125,
    (0, 2): 0.125,
    (1, 0): 0.6875,
    (1, 1): 0.5,
    (1, 2): 0.25,
    (2, 0): 0.875,
    (2, 1): 0.75,
    (2, 2): 0.5
}

def heuristic_value_func(state, hidden_state, player):
    good_win_payoff = 1.0 if hidden_state[player] in GOOD_ROLES else -float(state.NUM_GOOD)/state.NUM_EVIL
    if state.succeeds == 3:
        good_pass_three_prob = 1.0
    elif state.fails == 3:
        good_pass_three_prob = 0.0
    else:
        good_pass_three_prob = GOOD_PASS_THREE_PROB[(state.succeeds, state.fails)]

    find_merlin_prob = 1.0/state.NUM_GOOD
    good_win_prob = (1 - find_merlin_prob) * good_pass_three_prob + find_merlin_prob * 0.0
    return good_win_payoff*good_win_prob + (-good_win_payoff)*(1 - good_win_prob)



# Plays randomly, except always fails missions if bad.
class SingleMCTSPlayoutBot(Bot):
    def __init__(self):
        pass


    def reset(self, game, player, role, hidden_states):
        self.game = game
        self.player = player
        self.role = role
        self.hidden_states = hidden_states
        self.is_evil = role in EVIL_ROLES



    def handle_transition(self, old_state, new_state, observation, move=None):
        if old_state.status == 'run':
            if move is not None and self.role in EVIL_ROLES and not move.fail:
                observation += 1
            self.hidden_states = filter_hidden_states(self.hidden_states, old_state.proposal, observation)


    def get_action(self, state, legal_actions):
        return search_mcts(state, self.player, self.hidden_states)


    def get_move_probabilities(self, state, legal_actions):
        probs = np.zeros(len(legal_actions))
        probs[legal_actions.index(self.get_action(state, legal_actions))] = 1.0
        return probs


# Plays randomly, except always fails missions if bad.
class SingleMCTSHeuristicBot(Bot):
    def __init__(self):
        pass


    def reset(self, game, player, role, hidden_states):
        self.game = game
        self.player = player
        self.role = role
        self.hidden_states = hidden_states
        self.is_evil = role in EVIL_ROLES



    def handle_transition(self, old_state, new_state, observation, move=None):
        if old_state.status == 'run':
            if move is not None and self.role in EVIL_ROLES and not move.fail:
                observation += 1
            self.hidden_states = filter_hidden_states(self.hidden_states, old_state.proposal, observation)


    def get_action(self, state, legal_actions):
        return search_mcts(state, self.player, self.hidden_states, node_value_func=heuristic_value_func)


    def get_move_probabilities(self, state, legal_actions):
        probs = np.zeros(len(legal_actions))
        probs[legal_actions.index(self.get_action(state, legal_actions))] = 1.0
        return probs


class SingleMCTSBaseOpponentBot(Bot):
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
        hidden_state = self.hidden_states[np.random.choice(len(self.hidden_states))]
        _, probs = get_opponent_moves_and_probs(state, hidden_state, self.player, no_tremble=True)
        return probs
