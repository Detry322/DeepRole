import json
import glob
import numpy as np
import pandas as pd
import os
from collections import defaultdict

def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

def compute_csv_data(game):
    roles = game['game_info']['roles']
    roles = roles[:1] + roles[1:][::-1]
    if set(roles) != set(['Resistance', 'Merlin', 'Assassin', 'Spy']):
        return []
    res_bots = 0
    spy_bots = 0
    results = []
    game_id = os.urandom(4).encode('hex')
    for (player, role) in zip(game['session_info']['players'], roles):
        if 'DeepRole#' in player:
            if role in ['Resistance', 'Merlin']:
                res_bots += 1
            else:
                spy_bots += 1


    for index, (player, role) in enumerate(zip(game['session_info']['players'], roles)):
        is_resistance = role in ['Resistance', 'Merlin']
        resistance_win = game['game_info']['winner'] != 'Spy'
        results.append({
            'seat': index,
            'game': game_id,
            'num_bots': res_bots + spy_bots,
            'res_bots': res_bots,
            'spy_bots': spy_bots,
            'is_bot': 'DeepRole#' in player,
            'is_resistance': is_resistance,
            'role': role,
            'resistance_win': resistance_win,
            'win': not (resistance_win ^ is_resistance),
            'payoff': (1.0 if resistance_win else -1.0) * (1.0 if is_resistance else -1.5)
        })
    return results


def all_equal(l):
    for a in l:
        for b in l:
            if a != b:
                return False
    return True


def get_dataframe():
    games = defaultdict(lambda: {})
    for game_file in glob.glob('results_dir/*/*/*.json'):
        with open(game_file) as f:
            g = json.load(f)

            key = (
                tuple(g['session_info']['players']),
                g['game_info']['roomId'],
                tuple(g['game_info']['roles']),
                g['game_info']['missionNum'],
                g['game_info']['teamLeader'],
                tuple(g['game_info']['proposedTeam']),
                g['game_info']['winner'],
                to_tuple(g['game_info']['voteHistory'][g['session_info']['players'][0]]),
                to_tuple(g['game_info']['voteHistory'][g['session_info']['players'][1]]),
                to_tuple(g['game_info']['voteHistory'][g['session_info']['players'][2]]),
                to_tuple(g['game_info']['voteHistory'][g['session_info']['players'][3]]),
                to_tuple(g['game_info']['voteHistory'][g['session_info']['players'][4]]),
            )
            games[key][game_file] = g

    dataframe_data = []
    for _, similar_games in games.items():
        filenames = similar_games.keys()
        result = compute_csv_data(similar_games[filenames[0]])
        dataframe_data.extend(result)

    df = pd.DataFrame(dataframe_data, columns=[
        'game', 'num_bots', 'is_bot', 'win', 'payoff', 'is_resistance',
        'role', 'resistance_win', 'res_bots', 'spy_bots', 'seat'
    ])

    add_other_dataframes = pd.concat([ df, pd.read_csv('human_v_human_data.csv'), pd.read_csv('bot_v_bot_data.csv')])
    add_other_dataframes.reset_index(drop=True, inplace=True)
    return add_other_dataframes


def compute_prob(data):
    values, counts = np.unique(data, return_counts=True)
    if 0 in list(counts):
        return float('nan')
    better_counts = 0.5*np.ones(len(counts)) + counts
    samples = np.random.dirichlet(better_counts, 1000000)
    return np.mean((np.dot(samples, values) > 0.0).astype(np.float))



def calculate_aggregate_statistics(df):
    print "============================ Aggregate Stats =================================="
    print ""
    print " N_Bots | N_Humans | Bot_payoff | human_payoff | N_games | P(bot_payoff > 0.0) "
    print "-------------------------------------------------------------------------------"
    total_games = 0
    for num_bots in range(6):
        df_by_game = df[df.num_bots == num_bots]
        bot_payoffs = df_by_game[df_by_game.is_bot].payoff
        bot_avg = bot_payoffs.mean() if len(bot_payoffs) > 0 else 0.0
        human_avg = -bot_avg * (5.0 - num_bots) / num_bots if num_bots > 0 else df_by_game.payoff.mean()
        if num_bots == 5:
            human_avg = 0.0

        num_games = len(df_by_game.game.unique())
        total_games += num_games

        print " {: <6} | {: <8} | {: >10.05f} | {: >12.05f} | {: <7} | {}".format(
            num_bots,
            5 - num_bots,
            bot_avg,
            human_avg,
            num_games,
            compute_prob(bot_payoffs) if num_bots not in [0, 5] else 'undefined'
        )

    print "---------------------------------------------- | {: <7} | ---------------------".format(total_games)


from scipy.special import betaln
def confidence_b_gr_a(pos_a, neg_a, pos_b, neg_b):
    alpha_a = pos_a + 1
    beta_a = neg_a + 1
    alpha_b = pos_b + 1
    beta_b = neg_b + 1
    
    total = 0.0
    for i in range(alpha_b):
        total += np.exp(
            betaln(alpha_a + i, beta_b + beta_a)
            - np.log(beta_b + i)
            - betaln(1 + i, beta_b)
            - betaln(alpha_a, beta_a)
        )
    return total


def calculate_statistics_by_role(df):
    print "============================ Stats by role =================================="
    print ""
    # print compare_humans_and_bots(df, ['num_bots', 'is_resistance'])


def main():
    df = get_dataframe()
    df.to_csv('human_v_bots_data.csv', index=False)
    calculate_aggregate_statistics(df)
    calculate_statistics_by_role(df)

if __name__ == '__main__':
    main()

