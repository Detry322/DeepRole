import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Memory
memory = Memory('/tmp/joblib', verbose=1)

from data import (
    load_tensorboard_held_out_loss_data,
    load_tensorboard_loss_data,
    load_tensorboard_final_losses,
    TOPOLOGICALLY_SORTED_LAYERS,
    load_bot_v_bot_games,
    load_human_v_bot_games
)
from game_analysis import (
    filter_bot_v_bot_games,
    compute_payoff_matrix,
    calc_new_strat,
    compare_humans_and_bots,
    convert_to_by_player,
)

sns.set(
    style='ticks',
    context='notebook',
    font_scale = 1
)
matplotlib.rcParams['font.family'] = 'monospace'

print 'start'

@memory.cache
def cache_it():
    unconstrained_loss_data = load_tensorboard_final_losses('uncon_zero')
    unconstrained_loss_data['type'] = 'No Win Layer'
    improved_loss_data = load_tensorboard_final_losses('improved_arch')
    improved_loss_data['type'] = 'DeepRole'
    uncon_nozero = load_tensorboard_final_losses('uncon_nozero')
    uncon_nozero['type'] = 'No Win Layer\nNo Deduction'
    
    return pd.concat([unconstrained_loss_data, improved_loss_data, uncon_nozero]) 

df = cache_it()
order = ['DeepRole', 'No Win Layer',
         # 'No Win Layer\nNo Deduction'
]
df['succeeds'] = df.part.apply(lambda p: p[0])
df['fails'] = df.part.apply(lambda p: p[1])
df['count'] = df.part.apply(lambda p: p[2]+1)

fig, axes = plt.subplots(3,3, sharex=True, sharey=True, figsize=(8,8))

for s, f in [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (2, 0), (2, 1), (1, 2), (2, 2)]:
    ax = axes[s][f]
    g = sns.pointplot(x='count', y="loss", hue="type", data=df[(df.succeeds == s) & (df.fails == f)],
                      ax=axes[s][f], hue_order=order, join=False)
    g.set_title('succeed:{} fail:{}'.format(s,f))
    ax.set(
        ylim=[-0.0001, 0.001],
        # yscale = 'log'
    )
    # ax.set_yscale('log')
    if s+f != 0:
        ax.get_legend().remove()
    else:
        ax.get_legend().set_title('')
    
    if s == 2:
        ax.set_xlabel('proposal #')
    else:
        ax.set_xlabel('')

    if f == 0:
        ax.set_ylabel('MSE')
    else:
        ax.set_ylabel('')
        

# path = ''
path = '/Users/Max/Dropbox/Apps/Overleaf/DeepRole/figures/'

sns.despine()
plt.tight_layout()
plt.savefig(path + 'loss.pdf') ; plt.close()

sns.set(
    style='ticks',
    context='notebook',
    font_scale = 1.1
)
matplotlib.rcParams['font.family'] = 'monospace'
g = sns.catplot(x='type', y="loss", data=df, kind='bar', order=order, aspect=1.2, height=3.3, ci=68).set_axis_labels('', 'MSE')
plt.tight_layout()
plt.savefig(path + 'loss_mean.pdf'); plt.close()
    

print 'done'
