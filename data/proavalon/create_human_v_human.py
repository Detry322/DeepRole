import json
import glob
import numpy as np
import pandas as pd
import os
import sys
from collections import defaultdict

def parse_game(game):
    if game['gameMode'] != 'avalon':
        return []
    if set(game['roles']) != set(['Merlin', 'Assassin']):
        return []
    if len(game['playerRoles']) != 5:
        return []

    resistance_win = game['winningTeam'] == 'Resistance'

    results = []
    game_id = os.urandom(4).encode('hex')
    for player, role_data in game['playerRoles'].items():
        role = role_data['role']
        is_resistance = role in ['Merlin', "Resistance"]
        results.append({
            'seat': -1,
            'game': game_id,
            'num_bots': 0,
            'res_bots': 0,
            'spy_bots': 0,
            'is_bot': False,
            'is_resistance': is_resistance,
            'role': role,
            'resistance_win': resistance_win,
            'win': not (resistance_win ^ is_resistance),
            'payoff': (1.0 if resistance_win else -1.0) * (1.0 if is_resistance else -1.5)
        })
    return results


def main():
    if len(sys.argv) != 2:
        raise Exception("didn't pass a file")

    with open(sys.argv[1], 'r') as f:
        json_data = json.load(f)

    result = []

    for game in json_data:
        result.extend(parse_game(game))

    df = pd.DataFrame(result, columns=[
        'game', 'num_bots', 'is_bot', 'win', 'payoff', 'is_resistance',
        'role', 'resistance_win', 'res_bots', 'spy_bots', 'seat'
    ])

    df.to_csv('human_v_human_data.csv', index=False)

if __name__ == "__main__":
    main()

