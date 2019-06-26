import random
import numpy as np
import itertools

from battlefield.bots.bot import Bot
from battlefield.avalon_types import filter_hidden_states, EVIL_ROLES, GOOD_ROLES, VoteAction, ProposeAction, MissionAction

from collections import defaultdict

class ObserveBot(Bot):
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


    def get_action(self, state, legal_actions, role_guess=None, return_all=False):
        role_guess = role_guess or random.choice(self.hidden_states)
        if state.status == 'vote':
            if state.propose_count == 4:
                return VoteAction(up=True)

            up_vote = role_guess[state.proposer] in GOOD_ROLES and all(role_guess[p] in GOOD_ROLES for p in state.proposal)
            if self.is_evil:
                up_vote = not up_vote
            return VoteAction(up=up_vote)

        if state.status == 'propose' and not self.is_evil:
            propose_size = len(legal_actions[0].proposal)
            good_players = [ p for p, role in enumerate(role_guess) if role in GOOD_ROLES ]
            if return_all:
                return [ProposeAction(proposal=combo) for combo in itertools.combinations(good_players, propose_size)]
            random.shuffle(good_players)
            return ProposeAction(proposal=tuple(sorted(good_players[:propose_size])))

        if state.status == 'run' and self.is_evil:
            return MissionAction(fail=True)

        if return_all:
            return legal_actions
        return random.choice(legal_actions)


    def get_move_probabilities(self, state, legal_actions):
        move_weights = defaultdict(lambda: 0)
        for role_guess in self.hidden_states:
            actions = self.get_action(state, legal_actions, role_guess=role_guess, return_all=True)
            if not isinstance(actions, list):
                actions = [actions]
            for action in actions:
                move_weights[action] += 1.0 / len(actions)

        result = np.zeros(len(legal_actions))

        for move, weight in move_weights.items():
            result[legal_actions.index(move)] = weight

        return result / np.sum(result)


import itertools
def most_likely_team(team_probabilities, hidden_states):
    lls = []
    for assignment in hidden_states:
        ll = 0.0
        for i in range(len(assignment)):
            for j in range(i+1, len(assignment)):
                i_good = assignment[i] in GOOD_ROLES
                j_good = assignment[j] in GOOD_ROLES
                same_team = i_good == j_good
                ll += np.log(team_probabilities[i][j] if same_team else (1.0 - team_probabilities[i][j]))
        lls.append(ll)
    return max(zip(lls, hidden_states))


def update_pairings(votes, team_probabilities):
    for i, vote_i in enumerate(votes):
        for j, vote_j in enumerate(votes):
            pr_o_same_team = 0.6 if vote_i == vote_j else 0.4
            pr_o_diff_team = 0.4 if vote_i == vote_j else 0.6
            pr_o = team_probabilities[i][j] * pr_o_same_team + (1.0 - team_probabilities[i][j]) * pr_o_diff_team
            team_probabilities[i][j] = pr_o_same_team * team_probabilities[i][j] / pr_o


class ExamineAgreementBot(Bot):
    def __init__(self):
        self.bot = ObserveBot()

    def reset(self, *args):
        self.bot.reset(*args)
        self.team_probabilities = np.ones((len(args[3][0]), len(args[3][0]))) / 2


    def handle_transition(self, *args, **kwargs):
        if isinstance(args[2], tuple) and isinstance(args[2][0], VoteAction):
            update_pairings(args[2], self.team_probabilities)
        print most_likely_team(self.team_probabilities, self.bot.hidden_states)
        self.bot.handle_transition(*args, **kwargs)


    def get_action(self, state, legal_actions, role_guess=None, return_all=False):
        return self.bot.get_action(state, legal_actions, role_guess, return_all)


    def get_move_probabilities(self, state, legal_actions):
        return self.bot.get_move_probabilities(state, legal_actions)   
