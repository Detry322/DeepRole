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
    font_scale = 1.1
)
matplotlib.rcParams['font.family'] = 'monospace'

# path = ''
path = '/Users/Max/Dropbox/Apps/Overleaf/DeepRole/figures/'

df = pd.read_csv('data/held_out_loss_data.csv')
df.loc[df['model'] == 'win_probs', 'model'] = 'DeepRole'
df.loc[df['model'] == 'unconstrained', 'model'] = 'No Win Layer'
df.loc[:, 'n_datapoints'] = (df.loc[:, 'n_datapoints']/1000).astype(int)
g = sns.catplot(x='n_datapoints', y='held_out_loss', hue='model', kind='point', data=df, legend_out=False, aspect=1.1 , height=3.3, ci=68)
# g.axes.flat[-1].legend(frameon=False)
leg = g.axes.flat[0].get_legend()
leg.set_title('')

g.set_axis_labels('Training samples (1000)', 'MSE')
plt.savefig(path + 'sample.pdf')

print 'done'

