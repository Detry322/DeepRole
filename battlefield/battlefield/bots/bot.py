"""Base bot class"""
class Bot:
    def __init__(self):
        pass

    @classmethod
    def create_and_reset(cls, game, player, role, hidden_states):
        bot = cls()
        bot.reset(game, player, role, hidden_states)
        return bot


    def show_roles(self, hidden_state, bot_ids):
        pass


    def reset(self, game, player, role, hidden_states):
        pass


    def set_bot_ids(self, bot_ids):
        pass


    def handle_transition(self, old_state, new_state, observation, move=None):
        pass


    def get_action(self, state, legal_actions):
        raise NotImplemented


    def get_move_probabilities(self, state, legal_actions):
        raise NotImplemented
