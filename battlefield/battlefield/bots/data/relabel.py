import json
import os

DATAFILE = os.path.abspath(os.path.join(os.path.dirname(__file__), 'games.json'))
OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'relabeled.json'))

with open(DATAFILE, 'r') as f:
    input_json = json.load(f)

def relabel_log(log_item, first_proposer, num_players):
    new_log = {}

    for key in log_item:
        if key in set(['1', '2', '3', '4', '5']):
            new_log[key] = {
                'proposer': (2*num_players + log_item[key]['proposer'] - first_proposer) % num_players,
                'votes': log_item[key]['votes'][first_proposer:] + log_item[key]['votes'][:first_proposer],
                'team': [
                    (2*num_players + player - first_proposer) % num_players
                    for player in log_item[key]['team']
                ]
            }
        elif key == 'mission':
            new_log[key] = log_item[key]
        elif key in ['findMerlin', 'ladyOfTheLake']:
            new_log[key] = { k: (2*num_players + p - first_proposer) % num_players for k, p in log_item[key].items() }
        else:
            assert False, key
    return new_log


def relabel_players(game):
    num_players = len(game['players'])
    first_proposer = (game['roles']['leader'] + 1) % num_players
    if first_proposer == 0:
        return game

    output_game = {
        'end': game['end'],
        'log': [],
        'roles': {},
        'players': [],
        'start': game['start'],
        'spies_win': game['spies_win'],
        'id': game['id']
    }

    for role, person in game['roles'].items():
        output_game['roles'][role] = (2*num_players + person - first_proposer) % num_players

    for player in game['players']:
        output_game['players'].append({
            'player_id': player['player_id'],
            'spy': player['spy'],
            'seat': (2*num_players + player['seat'] - first_proposer) % num_players
        })
    output_game['players'].sort(key=lambda p: p['seat'])
    output_game['log'] = [ relabel_log(log_item, first_proposer, num_players) for log_item in game['log'] ]
    return output_game


output_json = [ relabel_players(game) for game in input_json ]

with open(OUTPUT, 'w') as f:
    json.dump(output_json, f, indent=2, sort_keys=True)
