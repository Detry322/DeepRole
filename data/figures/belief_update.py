import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Memory
memory = Memory('/tmp/joblib', verbose=1)

from data import (
    load_bot_v_bot_games,
    load_human_v_bot_games
)
from game_analysis import (
    filter_bot_v_bot_games,
    compute_payoff_matrix,
    calc_new_strat,
    compare_humans_and_bots,
    convert_to_by_player,
    create_moawt,
    moawt
)

sns.set(
    style='ticks',
    context='poster',
    font_scale = 1
)
matplotlib.rcParams['font.family'] = 'monospace'

def create_plots_for_data(big_df, ax, group_missions=False, opacity='22', linewidth=2.5, name=''):



    palette = ['#00FF00', '#FF0000', '#0000FF']
    big_df.loc[big_df['win_type']=='three missions passed', 'win_type'] = '3 passed'
    big_df.loc[big_df['win_type']=='three missions failed', 'win_type'] = '3 failed'

    big_df.sort_values('win_type', inplace=True)
    if group_missions:
        big_df.loc[big_df.win_type == 'merlin picked', 'color'] = 'blue'
        big_df.loc[big_df.win_type == 'merlin picked', 'win_type'] = '3 passed'
        palette = ['#0000FF', '#FF0000']

    g = sns.pointplot(x='ind', y='score', hue='win_type', hue_order = ['3 passed', '3 failed'],
                  data=big_df[(big_df.ind.round() == big_df.ind)], ci=68, 
                      join=False, palette=palette, ax = ax, legend=False)
    
    g.set(ylim=[-0.03,1])

    for game_id in big_df.game.unique():
        df = big_df[big_df.game == game_id].reset_index()
        color = df.loc[0]['color']
        sns.lineplot(x='ind', y='score', data=df, linewidth=linewidth, color={
            'red': '#FF0000' + opacity,
            'blue': '#0000FF' + opacity,
            'green': '#00FF00' + opacity
        }[color], legend=False, ax = ax)
    plt.xticks(
        np.arange(0, 5, 1),
        [
            'R1',   #'', '', '', '',
            'R2', #'2.1', '2.2', '2.3', '2.4',
            'R3', #'3.1', '3.2', '3.3', '3.4',
            'R4', #'4.1', '4.2', '4.3', '4.4',
            'R5', #'5.1', '5.2', '5.3', '5.4',
        ],
        rotation=0
    )
    # plt.gca().tick_params(axis='x', which='minor')

print 'start'
fig, axs = plt.subplots(1,2, sharey = True,figsize=(10,5), sharex= True)
    
create_plots_for_data(pd.read_csv('belief_data/4h1b_resistance.csv'), group_missions=True, opacity='50', linewidth=2, name= '4h1b', ax=axs[0])

create_plots_for_data(pd.read_csv('belief_data/5h_resistance.csv'), group_missions=True, opacity='20', linewidth=1, name = '5h', ax=axs[1])
for i in range(2):
    axs[i].get_legend().set_visible(False)
    axs[i].set_xlabel('Game stage (round)')
    axs[i].set_ylabel('Pr(true spy roles)')
sns.despine()
plt.tight_layout()
axs[1].legend(bbox_to_anchor=(-.85, .95), ncol=2, loc=3, frameon=False)
path = '/Users/Max/Dropbox/Apps/Overleaf/DeepRole/figures/'
plt.savefig(path+'belief.pdf'); plt.close()


print 'done'
