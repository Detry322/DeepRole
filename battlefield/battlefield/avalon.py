import itertools

from game import GameState
from battlefield.avalon_types import AVALON_PROPOSE_SIZES, AVALON_PLAYER_COUNT, EVIL_ROLES, GOOD_ROLES, ProposeAction, VoteAction, MissionAction, PickMerlinAction

class AvalonState(GameState):
    NUM_PLAYERS = None
    NUM_GOOD = None
    NUM_EVIL = None
    MISSION_SIZES = None

    def __init__(self, proposer, propose_count, succeeds, fails, status, proposal, game_end, num_players):
        self.NUM_PLAYERS = num_players
        self.NUM_GOOD, self.NUM_EVIL = AVALON_PLAYER_COUNT[num_players]
        self.MISSION_SIZES = AVALON_PROPOSE_SIZES[num_players]
        assert 0 <= proposer < self.NUM_PLAYERS, "Proposer invalid"
        assert 0 <= propose_count < 5, "Propose count invalid"
        assert 0 <= succeeds <= 3, "succeeds invalid"
        assert 0 <= fails <= 3, "fails invalid"
        assert 0 <= succeeds + fails <= 5, "Round invalid"
        assert status in set(['propose', 'vote', 'run', 'merlin', 'end']), "invalid status"
        assert status != 'end' or game_end is not None, "game end not consistent"
        assert status != 'merlin' or succeeds == 3, "merlin guess not consistent"
        assert status != 'run' or proposal is not None, "bad proposal for running"
        assert status != 'run' or len(proposal) == self.MISSION_SIZES[succeeds + fails][0]
        assert status != 'propose' or proposal is None, "proposal should be None before proposal"
        self.proposer = proposer
        self.propose_count = propose_count
        self.succeeds = succeeds
        self.fails = fails
        self.status = status
        self.proposal = proposal
        self.game_end = game_end


    def as_key(self):
        return (self.proposer, self.propose_count, self.succeeds, self.fails, self.status, self.proposal, self.game_end)


    @classmethod
    def start_state(cls, num_players):
        """
        Returns the starting state for a certain number of players
        """
        return cls(proposer=0, propose_count=0, succeeds=0, fails=0, status='propose', proposal=None, game_end=None, num_players=num_players)


    def new(self, *args, **kwargs):
        kwargs['num_players'] = self.NUM_PLAYERS
        return self.__class__(*args, **kwargs)


    def is_terminal(self):
        """
        Returns true if the game has ended
        """
        return self.status == 'end'


    def terminal_value(self, hidden_state):
        """
        Returns the payoff for each player
        """
        assert self.game_end[0] in set(['evil', 'good']), "Bad game end"
        # This ensures zero-sum

        num_good, num_evil = self.NUM_GOOD, self.NUM_EVIL

        good_amount = 1.0 if self.game_end[0] == 'good' else -1.0
        evil_amount = -float(num_good)/num_evil if self.game_end[0] == 'good' else float(num_good)/num_evil

        return [
            good_amount if player_role in GOOD_ROLES else evil_amount
            for player_role in hidden_state
        ]


    def moving_players(self):
        """
        Returns an array of players whose turn it is.
        """
        assert not self.is_terminal(), "Wat"
        if self.status == 'merlin':
            # It's the assassin guessing time
            return range(self.NUM_PLAYERS)

        if self.status == 'propose':
            return [self.proposer]

        if self.status == 'vote':
            return range(self.NUM_PLAYERS)

        if self.status == 'run':
            return sorted(list(self.proposal))

        assert False, "Not sure how we got here"



    def legal_actions(self, player, hidden_state):
        """
        Returns the legal actions of the player from this state, given a hidden state
        """
        assert player in self.moving_players(), "Asked a non-moving player legal actions"
        if self.status == 'merlin':
            return [PickMerlinAction(merlin=p) for p in range(self.NUM_PLAYERS)]

        if self.status == 'propose':
            proposal_size, _ = self.MISSION_SIZES[self.succeeds + self.fails]
            return [ProposeAction(proposal=p) for p in itertools.combinations(range(self.NUM_PLAYERS), r=proposal_size)]

        if self.status == 'vote':
            return [VoteAction(up=True), VoteAction(up=False)]

        if self.status == 'run':
            player_role = hidden_state[player]
            if player_role in EVIL_ROLES:
                return [MissionAction(fail=False), MissionAction(fail=True)]
            else:
                return [MissionAction(fail=False)]

        assert False, "Not sure how we got here"


    def vote_fail_transition(self):
        new_count = self.propose_count + 1
        new_proposer = (self.proposer + 1) % self.NUM_PLAYERS

        if new_count >= 5:
            return self.new(
                proposer=0,
                propose_count=0,
                succeeds=self.succeeds,
                fails=self.fails,
                status='end',
                proposal=None,
                game_end=('evil', 'Too many proposals and no resolution')
            )
        return self.new(
            proposer=new_proposer,
            propose_count=new_count,
            succeeds=self.succeeds,
            fails=self.fails,
            status='propose',
            proposal=None,
            game_end=None
        )


    def vote_pass_transition(self):
        return self.new(
            proposer=self.proposer,
            propose_count=self.propose_count,
            succeeds=self.succeeds,
            fails=self.fails,
            status='run',
            proposal=self.proposal,
            game_end=None
        )


    def vote_transition(self, hidden_state, votes):
        up_votes = sum([ 1 for vote in votes if vote.up ])
        if up_votes > self.NUM_PLAYERS/2:
            return self.vote_pass_transition(), hidden_state, tuple(votes)
        else:
            return self.vote_fail_transition(), hidden_state, tuple(votes)


    def mission_fail_transition(self):
        new_fails = self.fails + 1
        if new_fails == 3:
            return self.new(
                proposer=0,
                propose_count=0,
                succeeds=self.succeeds,
                fails=new_fails,
                status='end',
                proposal=None,
                game_end=('evil', 'Too many bad fails')
            )
        return self.new(
            proposer=(self.proposer + 1) % self.NUM_PLAYERS,
            propose_count=0,
            succeeds=self.succeeds,
            fails=new_fails,
            status='propose',
            proposal=None,
            game_end=None
        )


    def mission_succeed_transition(self):
        new_succeeds = self.succeeds + 1
        if new_succeeds == 3:
            return self.new(
                proposer=0,
                propose_count=0,
                succeeds=new_succeeds,
                fails=self.fails,
                status='merlin',
                proposal=None,
                game_end=None
            )
        return self.new(
            proposer=(self.proposer + 1) % self.NUM_PLAYERS,
            propose_count=0,
            succeeds=new_succeeds,
            fails=self.fails,
            status='propose',
            proposal=None,
            game_end=None
        )


    def mission_transition(self, hidden_state, mission_votes):
        _, num_fails_required = self.MISSION_SIZES[self.succeeds + self.fails]
        actual_fails = sum([ 1 for mission_vote in mission_votes if mission_vote.fail ])
        if actual_fails >= num_fails_required:
            return self.mission_fail_transition(), hidden_state, actual_fails
        else:
            return self.mission_succeed_transition(), hidden_state, actual_fails


    def proposal_transition(self, hidden_state, proposal):
        return self.new(
            proposer=self.proposer,
            propose_count=self.propose_count,
            succeeds=self.succeeds,
            fails=self.fails,
            status='vote',
            proposal=proposal,
            game_end=None
        ), hidden_state, proposal


    def pick_merlin_transition(self, hidden_state, assassin_picked_correctly):
        return self.new(
            proposer=self.proposer,
            propose_count=self.propose_count,
            succeeds=self.succeeds,
            fails=self.fails,
            status='end',
            proposal=None,
            game_end=('evil', 'assassin picked') if assassin_picked_correctly else ('good', 'assassin failed')
        ), hidden_state, assassin_picked_correctly


    def transition(self, moves, hidden_state):
        """
        Returns a tuple:
        state': the new state
        hidden_state': the new hidden state
        observation: the communal observation made by all of the players
        """
        assert len(moves) == len(self.moving_players()), "More players moved than allowed"
        # for player, move in zip(self.moving_players(), moves):
        #     assert move in self.legal_actions(player, hidden_state), '{}, {}, {}, {}'.format(move, self, player, hidden_state)

        if self.status == 'merlin':
            chosen_player = moves[hidden_state.index('assassin')].merlin
            assassin_picked_correctly = chosen_player == hidden_state.index('merlin')
            return self.pick_merlin_transition(hidden_state, assassin_picked_correctly)

        if self.status == 'propose':
            proposal = moves[0].proposal
            return self.proposal_transition(hidden_state, proposal)

        if self.status == 'vote':
            return self.vote_transition(hidden_state, moves)

        if self.status == 'run':
            return self.mission_transition(hidden_state, moves)

        assert False, "Not sure how we got here"


    def __repr__(self):
        return "<AvalonState " + " ".join("{}={}".format(*kv) for kv in sorted(self.__dict__.items())) + ">"


if __name__ == "__main__":
    state = AvalonState.start_state(5)
    hidden_state = ('merlin', 'minion', 'assassin', 'servant', 'servant')
    print state
    print hidden_state
    print state.legal_actions(0, hidden_state)
