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
    create_moawt
)

def to_str(a):
    return "{:0.1f}".format(a*100)

def se_diff_p(p1, n1, p2, n2):
    return np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)

def prettify_moawt(df):

    # Rename bots to be more pretty
    df.loc[df.loc[:,'Bot']=='0_ObserveBot', 'Bot']= 'LogicBot'
    df.loc[df.loc[:,'Bot']=='1_RandomBot', 'Bot']= 'Random'
    df.loc[df.loc[:,'Bot']=='2_CFRBot_6000000', 'Bot']= 'CFR'
    df.loc[df.loc[:,'Bot']=='3_ISMCTSBot', 'Bot']= 'ISMCTS'
    df.loc[df.loc[:,'Bot']=='4_MOISMCTSBot', 'Bot']= 'MOISMCTS'
    us_human = df['Bot'] == 'Human'

    # Calculate the Delta Win Rate
    delta_us_them = (df['_4us_us_winrate'] - df['_4us_them_winrate']).apply(to_str)
    
    se = se_diff_p(df['_4us_us_winrate'], df['_4us_us_n'] ,df['_4us_them_winrate'], df['_4us_them_n'])
    delta_us_them_se = delta_us_them + " + " + se.apply(to_str)

    
    delta_them_us = (df['_4them_us_winrate'] - df['_4them_them_winrate']).apply(to_str)

    se = se_diff_p(df['_4them_them_winrate'], df['_4them_them_n'] ,df['_4them_us_winrate'], df['_4them_us_n'])
    delta_them_us_se = delta_them_us + " + " + se.apply(to_str)

    df = pd.DataFrame.from_dict({
        'Bot': df.Bot,
        'Role': df.Role,
        'delta_us_them': np.where(us_human, delta_us_them_se, delta_us_them),
        'delta_them_us': np.where(us_human, delta_them_us_se, delta_them_us),
    })
    return  df.groupby(['Bot', 'Role']).first()
    

    
    # them_us_winrate = df['_4them_us_winrate'].apply(to_str)
    # them_them_winrate = df['_4them_them_winrate'].apply(to_str)
    # them_them_winrate_se = them_them_winrate + " + " + df['_4them_them_se'].apply(to_str)
    # us_us_winrate = df['_4us_us_winrate'].apply(to_str)
    # us_us_winrate_se = us_us_winrate + " + " + df['_4us_us_se'].apply(to_str)
    # us_them_winrate = df['_4us_them_winrate'].apply(to_str)
    # us_them_winrate_se = us_them_winrate + " + " + df['_4us_them_se'].apply(to_str)

    # import pdb; pdb.set_trace()
    
    # df = pd.DataFrame.from_dict({
    #     'Bot': df.Bot,
    #     'Role': df.Role,
    #     '1us_us': np.where(us_human, us_us_winrate_se, us_us_winrate),
    #     '2us_them': np.where(us_human, us_them_winrate_se, us_them_winrate),
    #     '3them_us': np.where(us_human, them_us_winrate_se, them_us_winrate),
    #     '4them_them': np.where(us_human, them_them_winrate_se, them_them_winrate),
    # })
    # df = df.groupby(['Bot', 'Role']).first()
    # df.columns = pd.MultiIndex.from_tuples([
    #     ('Deeprole Base', 'DR Winrate'), ('Deeprole Base', 'O Winrate'),
    #     ('Opponent Base', 'DR Winrate'), ('Opponent Base', 'O Winrate'),
    # ])
    # return df

BOT_V_BOT_GAMES = memory.cache(load_bot_v_bot_games)()
HUMAN_V_BOT_GAMES = memory.cache(load_human_v_bot_games)()

moawt = memory.cache(create_moawt)
result = moawt(BOT_V_BOT_GAMES, [
    'ObserveBot', 'RandomBot', 'CFRBot_6000000', 'ISMCTSBot', 'MOISMCTSBot'
])

print result

print prettify_moawt(result)

print prettify_moawt(result).to_latex()


