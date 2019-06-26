from battlefield.bots.ismcts.mcts_common import simulate, determinization_iterator
from battlefield.bots.ismcts.moismcts import Node, select_leaf, expand_if_needed, backpropagate
from battlefield.avalon_types import GOOD_ROLES, EVIL_ROLES


def search_mtmoismcts(searcher, initial_game_state, possible_hidden_states, num_iterations):
    all_roots = [
        [ Node(parent=None, incoming_edge=None), Node(parent=None, incoming_edge=None) ]
        for player in range(initial_game_state.NUM_PLAYERS)
    ]

    for i, initial_hidden_state in determinization_iterator(possible_hidden_states, num_iterations):
        roots = [ all_roots[player][role in EVIL_ROLES] for player, role in enumerate(initial_hidden_state) ]

        determinization = [(initial_game_state, initial_hidden_state)]
        nodes, determinization = select_leaf(roots, determinization)
        nodes, determinization = expand_if_needed(nodes, determinization)
        rewards = simulate(*determinization[-1])
        backpropagate(nodes, rewards, determinization)


    moves = []
    for depth, player in enumerate(initial_game_state.moving_players()):
        moves_per_hidden = []
        for is_evil in [False, True]:
            root = all_roots[player][is_evil]
            if len(root.children) == 0:
                moves_per_hidden.append(None)
                continue
            for _ in range(depth):
                root = root.children['_no_move']
            moves_per_hidden.append(max(root.children, key=lambda action: root.children[action].visit_count))
        moves.append(moves_per_hidden)

    return moves[initial_game_state.moving_players().index(searcher)], all_roots

