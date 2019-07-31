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
    style='white',
    context='notebook',
    font_scale = 2
)
matplotlib.rcParams['font.family'] = 'monospace'

print 'start'

def _calc_arrow(x, y, new_x_func):
    def _two_to_three(vec):
        return np.matmul([[0, 2.0/(3 ** 0.5)], [1, -1/(3 ** 0.5)], [-1, -1/(3 ** 0.5)]], vec) + np.array([0, 0.5, 0.5])


    def _three_to_two(vec):
        return np.matmul(np.array([[0.5, 1, 0], [1.5 * (3 ** 0.5), (3 ** 0.5), (3 ** 0.5)]]), vec - np.array([0, 0.5, 0.5]))
    
    if y < 0:
        return 0.0, 0.0
    if y > (3 ** 0.5) * (x + 0.5):
        return 0.0, 0.0
    if y > - (3 ** 0.5) * (x - 0.5):
        return 0.0, 0.0
    
    probs = _two_to_three(np.array([x, y]))
    
    new_probs = new_x_func(probs)
    new_xy = _three_to_two(new_probs)
    return new_xy[0] - x, new_xy[1] - y


def make_triangle_plot(labels, new_x_func):
    scale = 0.07
    x = np.arange(-0.5, 0.5, scale * (2 / (3 ** 0.5)))
    y = np.arange(0, (3 ** 0.5) / 2, scale)
    X, Y = np.meshgrid(x, y)
    for i in range(0, len(X), 2):
        X[i] += scale/(3 ** 0.5)
    U, V = np.meshgrid(x, y)
    
    for i in range(len(U)):
        for j in range(len(U[0])):
            U[i][j], V[i][j] = _calc_arrow(X[i][j], Y[i][j], new_x_func)
    
    color = np.sqrt(U * U + V * V)

    # plt.figure(figsize=(16,13))
    # plt.quiver(X, Y, U/color, V/color, color, scale=1.5/scale)
    plt.quiver(X, Y, U/color*color, V/color*color, color, cmap=plt.cm.viridis_r)
    plt.text(-0.5, -0.09 * (1+('\n' in labels[2])), labels[2], horizontalalignment='center')
    plt.text(0.4, -0.09, labels[1], horizontalalignment='center')
    plt.text(0, 0.85, labels[0], horizontalalignment='center')
    plt.gca().axis('equal')
    plt.gca().axis('off')
    plt.tight_layout()

@memory.cache
def cache_it(bots):
    BOT_V_BOT_GAMES = load_bot_v_bot_games()
    dr_games = filter_bot_v_bot_games(BOT_V_BOT_GAMES, bots)
    dr_payoffs = compute_payoff_matrix(dr_games, bots)

    return dr_payoffs

path = '/Users/Max/Dropbox/Apps/Overleaf/DeepRole/figures/'
# path = ''

dr_bots = ['Deeprole_100_50', 'Deeprole_30_15', 'Deeprole_10_5']
dr_payoffs = cache_it(dr_bots)
dr_bots = ['DeepRole100', 'DeepRole30', 'DeepRole10']
make_triangle_plot(dr_bots, lambda probs: calc_new_strat(probs, dr_payoffs))

plt.savefig(path + 'triDR.pdf'); plt.close()

hand_bots = ['Deeprole', 'ObserveBot', 'CFRBot_6000000']
hand_payoffs = cache_it(hand_bots)
hand_bots = ['DeepRole', 'LogicBot', 'CFR']
make_triangle_plot(hand_bots, lambda probs: calc_new_strat(probs, hand_payoffs))

plt.savefig(path + 'triHand.pdf'); plt.close()

@memory.cache
def cache_it_last():
    import gzip
    with gzip.open('data/lesion_games.msg.gz') as f:
        df = pd.read_msgpack(f)

    dr_bots = ['Deeprole_ZeroingWinProbs', 'Deeprole_ZeroingUnconstrained', 'Deeprole_NoZeroUnconstrained']
    dr_payoffs = compute_payoff_matrix(df, dr_bots)
    return dr_payoffs

dr_payoffs = cache_it_last()
make_triangle_plot( ['DeepRole', 'No Win Layer', 'No Win Layer \nNo Deduction '], lambda probs: calc_new_strat(probs, dr_payoffs))
plt.tight_layout()
plt.savefig(path + 'triLesion.pdf'); plt.close()


print 'end'
