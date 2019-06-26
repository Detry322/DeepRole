import random
import math
import itertools
import numpy as np
from collections import defaultdict

from battlefield.bots.ismcts.mcts_common import random_choice, determinization_iterator, simulate, fast_simulate

class Node:
    def __init__(self, parent, incoming_edge):
        self.parent = parent
        self.incoming_edge = incoming_edge
        self.children = {} # map from joint actions to child nodes
        self.total_reward = 0.0 # the total reward for the parent for selecting this action
        self.availability_count = 0
        self.visit_count = 0
        self.exp3_sum = defaultdict(lambda: defaultdict(lambda: 0.0)) # map from player to action to reward

    def compatible_children(self, game_state, hidden_state):
        moving_players = game_state.moving_players()
        return [
            moves
            for moves in itertools.product(*[
                game_state.legal_actions(player, hidden_state)
                for player in moving_players
            ])
        ]

    def unexplored_children(self, game_state, hidden_state):
        return [moves for moves in self.compatible_children(game_state, hidden_state) if moves not in self.children]


    def calculate_exp3_probs(self, game_state, hidden_state, player):
        available_actions = game_state.legal_actions(player, hidden_state)
        K = len(available_actions)
        n = self.visit_count
        if n == 0:
            gamma = 1.0
        else:
            gamma = min(1.0, math.sqrt(K * math.log(K)/(n*(math.e - 1))))

        eta = gamma/K
        probs = [
            (
                gamma/K +
                (1.0 - gamma)/sum([
                    math.exp(min(eta*(self.exp3_sum[player][a] - self.exp3_sum[player][action]), 700))
                    for a in available_actions
                ])
            )
            for action in available_actions
        ]
        return available_actions, probs


    def select_child(self, game_state, hidden_state):
        moving_players = game_state.moving_players()
        if len(moving_players) == 1:
            available_actions = self.compatible_children(game_state, hidden_state)
            if len(available_actions) == 1:
                return available_actions[0]

            return max(
                available_actions,
                # UCB1
                key=lambda action: (
                    (self.children[action].total_reward/self.children[action].visit_count) +
                    2000 * math.sqrt(math.log(self.children[action].availability_count)/self.children[action].visit_count)
                )
            )
        else:
            move = ()
            for player in moving_players:
                actions, probs = self.calculate_exp3_probs(game_state, hidden_state, player)
                if len(actions) == 1:
                    chosen_move = actions[0]
                else:
                    chosen_move = random_choice(actions, p=probs)
                move += (chosen_move, )
            return move


def select_leaf(node, game_state, hidden_state):
    if game_state.is_terminal():
        return node, game_state, hidden_state
    if len(node.unexplored_children(game_state, hidden_state)) != 0:
        return node, game_state, hidden_state

    action = node.select_child(game_state, hidden_state)
    new_node = node.children[action]
    new_game_state, new_hidden_state, _ = game_state.transition(action, hidden_state)
    return select_leaf(new_node, new_game_state, new_hidden_state)


def expand_if_needed(node, game_state, hidden_state):
    if game_state.is_terminal():
        return node, game_state, hidden_state
    unexplored_children = node.unexplored_children(game_state, hidden_state)
    
    action = random_choice(unexplored_children)

    new_node = Node(parent=node, incoming_edge=action)
    node.children[action] = new_node
    new_game_state, new_hidden_state, _ = game_state.transition(action, hidden_state)
    return new_node, new_game_state, new_hidden_state


def backpropagate(initial_game_state, initial_hidden_state, node, rewards):
    action_history = []
    while node.parent is not None:
        action_history.append(node.incoming_edge)
        node = node.parent
    action_history = action_history[::-1]

    game_state, hidden_state = initial_game_state, initial_hidden_state
    for action in action_history:
        moving_players = game_state.moving_players()
        for neighbor in node.compatible_children(game_state, hidden_state):
            if neighbor in node.children:
                node.children[neighbor].availability_count += 1

        node.children[action].visit_count += 1

        if len(moving_players) == 1:
            node.children[action].total_reward += rewards[moving_players[0]]
        else:
            for player, move in zip(moving_players, action):
                if move not in node.exp3_sum[player]:
                    node.exp3_sum[player][move] += rewards[player]
                else:
                    actions, probs = node.calculate_exp3_probs(game_state, hidden_state, player)
                    prob = probs[actions.index(move)]
                    node.exp3_sum[player][move] += rewards[player] / prob

        node = node.children[action]
        game_state, hidden_state, _ = game_state.transition(action, hidden_state)


def search_ismcts(searcher, initial_game_state, possible_hidden_states, num_iterations):
    root = Node(parent=None, incoming_edge=None)

    for i, initial_hidden_state in determinization_iterator(possible_hidden_states, num_iterations):
        node, game_state, hidden_state = select_leaf(root, initial_game_state, initial_hidden_state)
        node, game_state, hidden_state = expand_if_needed(node, game_state, hidden_state)
        rewards = fast_simulate(game_state, hidden_state)
        backpropagate(initial_game_state, initial_hidden_state, node, rewards)

    moves = max(root.children, key=lambda action: root.children[action].visit_count)
    move = moves[initial_game_state.moving_players().index(searcher)]
    return move, root
