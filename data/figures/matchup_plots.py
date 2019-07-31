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

@memory.cache
def cache_it_all():
    BOT_V_BOT_GAMES = memory.cache(load_bot_v_bot_games)()
    HUMAN_V_BOT_GAMES = memory.cache(load_human_v_bot_games)()

    matchups = [
        'ObserveBot',
        'CFRBot_6000000',
        'MOISMCTSBot',
        'ISMCTSBot',
        'RandomBot',
    ]

    df = moawt(BOT_V_BOT_GAMES, [
        'ObserveBot', 'RandomBot', 'CFRBot_6000000', 'ISMCTSBot', 'MOISMCTSBot'
    ])
    return df
df = cache_it_all()
# path = ''
path = '/Users/Max/Dropbox/Apps/Overleaf/DeepRole/figures/'

# sns.set_style("whitegrid")
# sns.set_context("poster",
#                # font_scale=1.5, rc={"lines.linewidth": 2.5}
# )

sns.set(
    style='ticks',
    context='poster',
    font_scale = 1
)

matplotlib.rcParams['font.family'] = 'monospace'
# matplotlib.rcParams['font.monospace'] = ['Menlo']
matplotlib.rcParams["legend.handletextpad"] = 0.

rename = {
    '0_ObserveBot': 'LogicBot',
    '1_RandomBot': 'Random',
    '2_CFRBot_6000000': 'CFR',
    '3_ISMCTSBot': 'ISMCTS',
    '4_MOISMCTSBot': 'MOISMCTS'
}

for b in rename:
    df.loc[(df['Bot'] == b), ('Bot')] = rename[b]
OG_df = df.copy()

team_rename = {
    'all':'',
    'spy':'S ',
    'res':'R '
}

leg_rename = {
    'all':'',
    'spy':' (S)',
    'res':' (R)'
}

df = OG_df.copy()
df.loc[(df['b'] == 'bot'), ('b')] = '+DeepRole'
df.loc[(df['b'] == 'human'), ('b')] = '+Other'
g = sns.catplot(x='DR', y='WinRate', hue='b', col='Bot', row='Role', data=df, kind='point', ci=0, legend=False,
                legend_out = False, aspect = 1.2, height=4,
                col_order = ['Random', 'LogicBot', 'ISMCTS', 'MOISMCTS', 'CFR'], row_order = ['all', 'spy', 'res'])

g.set_titles('')
for ax,t  in zip(g.axes[0], ['Random', 'LogicBot', 'ISMCTS', 'MOISMCTS', 'CFR']):
    ax.set_title(t)
    

for ax in g.axes[2]:
    ax.set_xlabel('# DeepRole')

g.axes[0][0].legend(frameon=False)
g.axes[0][0].set_ylabel('P(Win)')
g.axes[1][0].legend(frameon=False)
g.axes[1][0].get_legend().get_texts()[0].set_text('+DeepRole (S)')
g.axes[1][0].get_legend().get_texts()[1].set_text('+Other (S)')
g.axes[1][0].set_ylabel('P(S Win)')

g.axes[2][0].legend(frameon=False)
g.axes[2][0].get_legend().get_texts()[0].set_text('+DeepRole (R)')
g.axes[2][0].get_legend().get_texts()[1].set_text('+Other (R)')
g.axes[2][0].set_ylabel('P(R Win)')
plt.tight_layout()
plt.savefig(path + 'botvbot.pdf'.format(r)); plt.close()


# for r in ['all', 'spy', 'res']:
#     # fig, ax = plt.subplots(1, 5, figsize=(12,12))
#     # fig, ax = plt.subplots(1, 5, figsize=(12,12))
#     # plt.figure(figsize=(45,10))
#     # sns.set(rc={'figure.figsize':(12,12)})

#     df = OG_df.copy()
#     df.loc[(df['b'] == 'bot'), ('b')] = '+DeepRole{}'.format(leg_rename[r])
#     df.loc[(df['b'] == 'human'), ('b')] = '+Other{}'.format(leg_rename[r])
#     g = sns.catplot(x='DR', y='WinRate', hue='b', col='Bot', row='Role', data=df[df['Role']==r], kind='point', ci=0, legend=False,
#                     legend_out = True, aspect = 1, height=5,
#                     col_order = ['Random', 'LogicBot', 'ISMCTS', 'MOISMCTS', 'CFR'])
#     g.set(
#         ylim = [0, 1])
#     # if r == 'all':
#     g.set_titles('{col_name}')
#     # else:
#         # g.set_titles('')
        
#     g.set_axis_labels('# of DeepRole', 'P({}Win)'.format(team_rename[r]))
    
#     # leg = g._legend
#     g.axes.flat[-1].legend(frameon=False)
#     # leg = g.axes.flat[0].get_legend()
#     # leg.set_title('')
#     # plt.gcf().set_size_inches(7, 1.8)
#     plt.tight_layout()
#     plt.savefig(path + 'botvbot_{}.pdf'.format(r)); plt.close()

# import pdb; pdb.set_trace()

# crafted_bots = 
# df = filter_bot_v_bot_games(BOT_V_BOT_GAMES, crafted_bots)
# # df['DR_count'] = (
# #     (df['bot_0']=='Deeprole').astype(int) +
# #     (df['bot_1']=='Deeprole') +
# #     (df['bot_2']=='Deeprole') +
# #     (df['bot_3']=='Deeprole') +
# #     (df['bot_4']=='Deeprole') 
# # )

# # for i in range(5):
# #     df.loc[df['bot_{}_payoff'.format(i)] > 0, 'bot_{}_payoff'.format(i)] = 1
# #     df.loc[df['bot_{}_payoff'.format(i)] < 0, 'bot_{}_payoff'.format(i)] = 0
    
# crafted_payoffs = compute_payoff_matrix(df, crafted_bots)

# import pdb; pdb.set_trace()
