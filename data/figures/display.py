import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _two_to_three(vec):
    return np.matmul([[0, 2.0/(3 ** 0.5)], [1, -1/(3 ** 0.5)], [-1, -1/(3 ** 0.5)]], vec) + np.array([0, 0.5, 0.5])


def _three_to_two(vec):
    return np.matmul(np.array([[0.5, 1, 0], [1.5 * (3 ** 0.5), (3 ** 0.5), (3 ** 0.5)]]), vec - np.array([0, 0.5, 0.5]))


def _calc_arrow(x, y, new_x_func):
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
    scale = 0.03
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
    
    plt.figure(figsize=(16,13))
    plt.quiver(X, Y, U/color, V/color, color, scale=1.5/scale)
    plt.text(-0.5, -0.02, labels[2], horizontalalignment='center')
    plt.text(0.5, -0.02, labels[1], horizontalalignment='center')
    plt.text(0, 0.885, labels[0], horizontalalignment='center')

def to_str(a):
    return "{:0.1f}%".format(a*100)

def prettify_moawt(df):
    us_human = df['Bot'] == 'Human'

    them_us_winrate = df['_4them_us_winrate'].apply(to_str)
    them_us_winrate_se = them_us_winrate + " + " + df['_4them_us_se'].apply(to_str)
    them_them_winrate = df['_4them_them_winrate'].apply(to_str)
    them_them_winrate_se = them_them_winrate + " + " + df['_4them_them_se'].apply(to_str)
    us_us_winrate = df['_4us_us_winrate'].apply(to_str)
    us_us_winrate_se = us_us_winrate + " + " + df['_4us_us_se'].apply(to_str)
    us_them_winrate = df['_4us_them_winrate'].apply(to_str)
    us_them_winrate_se = us_them_winrate + " + " + df['_4us_them_se'].apply(to_str)

    df = pd.DataFrame.from_dict({
        'Bot': df.Bot,
        'Role': df.Role,
        '1us_us': np.where(us_human, us_us_winrate_se, us_us_winrate),
        '2us_them': np.where(us_human, us_them_winrate_se, us_them_winrate),
        '3them_us': np.where(us_human, them_us_winrate_se, them_us_winrate),
        '4them_them': np.where(us_human, them_them_winrate_se, them_them_winrate),
    })
    df = df.groupby(['Bot', 'Role']).first()
    df.columns = pd.MultiIndex.from_tuples([
        ('Deeprole Base', 'DR Winrate'), ('Deeprole Base', 'O Winrate'),
        ('Opponent Base', 'DR Winrate'), ('Opponent Base', 'O Winrate'),
    ])
    return df
