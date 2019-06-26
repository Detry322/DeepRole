class GameState:
    NUM_PLAYERS = 0
    HIDDEN_STATES = []

    @classmethod
    def start_state(self):
        """
        Returns the starting state
        """
        raise NotImplemented


    def is_terminal(self):
        """
        Returns true if the game has ended
        """
        raise NotImplemented


    def terminal_value(self, hidden_state):
        """
        Returns the payoff for each player
        """
        raise NotImplemented


    def moving_players(self):
        """
        Returns an array of players whose turn it is.
        """
        raise NotImplemented


    def legal_actions(self, player, hidden_state):
        """
        Returns the legal actions of the player from this state, given a hidden state
        """
        raise NotImplemented


    def transition(self, moves, hidden_state):
        """
        Returns a tuple:
        state': the new state
        hidden_state': the new hidden state
        observation: the communal observation made by all of the players
        """
        raise NotImplemented
