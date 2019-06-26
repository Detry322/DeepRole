import json
import glob
import numpy as np
import pandas as pd
import os
import sys
import glob
import gzip
from collections import defaultdict

role_to_role = {
    'servant': "Resistance",
    'merlin': "Merlin",
    'assassin': "Assassin",
    'minion': "Spy"
}

def analyze_game(game):
    game_id = os.urandom(4).encode('hex')
    results = []
    resistance_win = game['winner'] == 'good'
    for player in range(5):
        role = role_to_role[game['bot_{}_role'.format(player)]]
        is_resistance = role in ['Merlin', "Resistance"]
        results.append({
            'seat': player,
            'game': game_id,
            'num_bots': 5,
            'res_bots': 3,
            'spy_bots': 2,
            'is_bot': True,
            'is_resistance': is_resistance,
            'role': role,
            'resistance_win': resistance_win,
            'win': not (resistance_win ^ is_resistance),
            'payoff': (1.0 if resistance_win else -1.0) * (1.0 if is_resistance else -1.5)
        })
    return results


def main():
    if len(sys.argv) != 2:
        raise Exception("didn't pass a folder")

    data = []

    for filename in glob.glob(sys.argv[1] + '/*.msg.gz'):
        with gzip.open(filename, 'r') as f:
            data.append(pd.read_msgpack(f))

    df = pd.concat(data)
    df.reset_index(drop=True, inplace=True)

    result = []
    for i in range(len(df)):
        result.extend(analyze_game(df.loc[i]))

    result_df = pd.DataFrame(result, columns=[
        'game', 'num_bots', 'is_bot', 'win', 'payoff', 'is_resistance',
        'role', 'resistance_win', 'res_bots', 'spy_bots', 'seat'
    ])

    result_df.to_csv('bot_v_bot_data.csv', index=False)


if __name__ == "__main__":
    main()
