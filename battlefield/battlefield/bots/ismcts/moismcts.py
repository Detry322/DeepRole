import math

from battlefield.bots.ismcts.mcts_common import random_choice, determinization_iterator, simulate, fast_simulate

class Node:
    def __init__(self, parent, incoming_edge):
        self.parent = parent
        self.incoming_edge = incoming_edge
        self.children = {} # map from joint actions to child nodes
        self.total_reward = 0.0 # the total reward for the parent for selecting this action
        self.availability_count = 0
        self.visit_count = 0


    def compatible_children(self, player, game_state, hidden_state):
        assert player in game_state.moving_players(), "Player not allowed to move"
        return game_state.legal_actions(player, hidden_state)


    def unexplored_children(self, player, game_state, hidden_state):
        return [move for move in self.compatible_children(player, game_state, hidden_state) if move not in self.children]


    def select_child(self, player, game_state, hidden_state):
        assert player in game_state.moving_players(), "Player not allowed to move"

        available_actions = self.compatible_children(player, game_state, hidden_state)

        if len(available_actions) == 1:
            return available_actions[0]

        return max(
            available_actions,
            key=lambda action: (
                (self.children[action].total_reward/self.children[action].visit_count) +
                5 * math.sqrt(math.log(self.children[action].availability_count)/self.children[action].visit_count)
            )
        )


def has_unexplored_children(nodes, game_state, hidden_state):
    for depth, player in enumerate(game_state.moving_players()):
        node = nodes[player]
        for _ in range(depth):
            if '_no_move' not in node.children:
                return True
            node = node.children['_no_move']
        if len(node.unexplored_children(player, game_state, hidden_state)) != 0:
            return True
    return False


def traverse_move(nodes, player, move):
    new_nodes = []
    for p, node in enumerate(nodes):
        if p == player:
            new_nodes.append(node.children[move])
        else:
            if '_no_move' not in node.children:
                node.children['_no_move'] = Node(parent=node, incoming_edge='_no_move')
            new_nodes.append(node.children['_no_move'])
    return new_nodes


def traverse_observation(nodes, observation, create=True):
    if isinstance(observation, tuple) and hasattr(observation[0], 'up'):
        observation = len([o for o in observation if o.up]) > 2

    if create:
        for node in nodes:
            if observation not in node.children:
                node.children[observation] = Node(parent=node, incoming_edge=observation)

    return [node.children[observation] for node in nodes]

def select_leaf(nodes, determinization):
    game_state, hidden_state = determinization[-1]

    if game_state.is_terminal():
        return nodes, determinization
    if has_unexplored_children(nodes, game_state, hidden_state):
        return nodes, determinization

    moves = []
    for player in game_state.moving_players():
        player_action = nodes[player].select_child(player, game_state, hidden_state)
        moves.append(player_action)
        nodes = traverse_move(nodes, player, player_action)

    new_state, new_hidden, observation = game_state.transition(moves, hidden_state)
    determinization.append((new_state, new_hidden))
    
    nodes = traverse_observation(nodes, observation)

    return select_leaf(nodes, determinization)


def expand_if_needed(nodes, determinization):
    game_state, hidden_state = determinization[-1]

    if game_state.is_terminal():
        return nodes, determinization

    moves = []
    for player in game_state.moving_players():
        unexplored_children = nodes[player].unexplored_children(player, game_state, hidden_state)
        if len(unexplored_children) == 0:
            move = nodes[player].select_child(player, game_state, hidden_state)
        else:
            move = random_choice(unexplored_children)

        for p, node in enumerate(nodes):
            if player != p and '_no_move' not in node.children:
                node.children['_no_move'] = Node(parent=node, incoming_edge='_no_move')
            if player == p and move not in node.children:
                node.children[move] = Node(parent=node, incoming_edge=move)

        nodes = traverse_move(nodes, player, move)
        moves.append(move)

    new_state, new_hidden, observation = game_state.transition(moves, hidden_state)
    determinization.append((new_state, new_hidden))

    nodes = traverse_observation(nodes, observation)

    return nodes, determinization


def get_parent_move(nodes):
    is_barrier = not any('_no_move' == node.incoming_edge for node in nodes)
    if is_barrier:
        return is_barrier, None
    return is_barrier, next(node.incoming_edge for node in nodes if node.incoming_edge != '_no_move')


def backtrace_actions(nodes):
    actions = []
    current_move = []
    while nodes[0].parent is not None:
        is_barrier, parent_move = get_parent_move(nodes)
        if is_barrier:
            actions.append(current_move[::-1])
            current_move = []
        else:
            current_move.append(parent_move)

        nodes = [node.parent for node in nodes]

    assert all(node.parent is None for node in nodes), "bad traversal logic"
    actions.append(current_move[::-1])
    return actions[:0:-1], nodes


def backpropagate(nodes, rewards, determinization):
    actions, nodes = backtrace_actions(nodes)
    for action_list, (game_state, hidden_state) in zip(actions, determinization):

        for player, action in zip(game_state.moving_players(), action_list):
            node = nodes[player]
            node.children[action].visit_count += 1
            node.children[action].total_reward += rewards[player]
            for neighbor in node.compatible_children(player, game_state, hidden_state):
                if neighbor in node.children:
                    node.children[neighbor].availability_count += 1

            nodes = traverse_move(nodes, player, action)

        _, _, observation = game_state.transition(action_list, hidden_state)
        nodes = traverse_observation(nodes, observation, create=False)



def search_moismcts(searcher, initial_game_state, possible_hidden_states, num_iterations):
    roots = [ Node(parent=None, incoming_edge=None) for player in range(initial_game_state.NUM_PLAYERS) ]

    for i, initial_hidden_state in determinization_iterator(possible_hidden_states, num_iterations):
        determinization = [(initial_game_state, initial_hidden_state)]
        nodes, determinization = select_leaf(roots, determinization)
        nodes, determinization = expand_if_needed(nodes, determinization)
        rewards = fast_simulate(*determinization[-1])
        backpropagate(nodes, rewards, determinization)

    moves = []
    for depth, player in enumerate(initial_game_state.moving_players()):
        root = roots[player]
        for _ in range(depth):
            root = root.children['_no_move']
        moves.append(max(root.children, key=lambda action: root.children[action].visit_count))
    return moves[initial_game_state.moving_players().index(searcher)], roots
