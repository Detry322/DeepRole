#!/usr/bin/env python
# coding: utf-8

# # Figures for the paper
# 
# Most figure generation code is in `display.py`, data loading code in `data.py`.
# 
# ## Part 1: Training figures
# 
# - [ ] Training curves as a function of data
# - [x] Last layer training loss for old and new model
# - [x] Second to last layer training loss for old and new model
# - [x] Total loss by layer (old vs. new)
# 
# ## Part 2: Bot vs. Bot Comparisons
# 
# - [x] DR100 vs DR30 vs DR10 [DR vs. itself]
#     - [x] Triangle chart
#     - [x] Payoff table
# - [x] DR100 vs CFR6K vs ObserveBot [DR vs. hand crafted methods]
#     - [x] Triangle chart [pending DeepStack todo]
#     - [x] Payoff table [pending DeepStack todo]
# - [x] DR100 vs DeepStack + MOISMCTS [DR vs. existing methods]
#     - [ ] Payoff table
#     
# ## Part 3: Human vs. Bot Comparisons
# 
# - [x] Win rate table
#     - [x] Overall
#     - [x] By role
# - [ ] Wellman analysis
#     - [ ] Payoff Table
#     - [ ] Line chart

# In[1]:

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)
import numpy as np
import pandas as pd
import seaborn as sns

from display import (
    make_triangle_plot,
    prettify_moawt
)
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
    create_moawt
)


# # Part 1: Training figures

# ## 1.1 - Training curves as a function of data

# In[12]:


x = load_tensorboard_held_out_loss_data('good_model')
for v in x.values():
    print min(v)
print "----"
x = load_tensorboard_held_out_loss_data('bad_model')
for v in x.values():
    print min(v)


# ## 1.2 - Last layer training loss for old and new model

# In[22]:


unconstrained_loss_data = load_tensorboard_loss_data('unconstrained_arch', *TOPOLOGICALLY_SORTED_LAYERS[-1]) # Last layer
improved_loss_data = load_tensorboard_loss_data('improved_arch', *TOPOLOGICALLY_SORTED_LAYERS[-1]) # Last layer

sns.lineplot(x=range(3000), y=unconstrained_loss_data, label='Zero-sum layer')
sns.lineplot(x=range(3000), y=improved_loss_data, label='Improved Architecture')
plt.ylim(0, 0.004)
plt.title("Validation loss per epoch (last training layer)")
plt.ylabel("Validation MSE loss")
plt.xlabel("Training epoch")
plt.show()


# ## 1.3 - Second to last layer training loss for old and new model

# In[23]:


unconstrained_loss_data = load_tensorboard_loss_data('unconstrained_arch', *TOPOLOGICALLY_SORTED_LAYERS[-2]) # Penultimate layer
improved_loss_data = load_tensorboard_loss_data('improved_arch', *TOPOLOGICALLY_SORTED_LAYERS[-2]) # Penultimate layer

sns.lineplot(x=range(3000), y=unconstrained_loss_data, label='Zero-sum layer')
sns.lineplot(x=range(3000), y=improved_loss_data, label='Improved Architecture')
plt.ylim(0, 0.004)
plt.title("Validation loss per epoch (second-to-last training layer)")
plt.ylabel("Validation MSE loss")
plt.xlabel("Training epoch")
plt.show()


# ## 1.4 - Total loss by layer

# In[20]:


unconstrained_loss_data = load_tensorboard_final_losses('unconstrained_arch')
unconstrained_loss_data['type'] = 'unconstrained'
improved_loss_data = load_tensorboard_final_losses('improved_arch')
improved_loss_data['type'] = 'improved'

df = pd.concat([unconstrained_loss_data, improved_loss_data])
df['succeeds'] = df.part.apply(lambda p: p[0])
df['fails'] = df.part.apply(lambda p: p[1])
df['count'] = df.part.apply(lambda p: p[2])

for s, f in [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (2, 0), (2, 1), (1, 2), (2, 2)]:
    sns.barplot(x='count', y="loss", hue="type", data=df[(df.succeeds == s) & (df.fails == f)])
    plt.show()


# ## 1.5 - Loss by distance from end

# In[21]:


# is this even well formed


# # Part 2: Bot vs. Bot Comparisons

# In[6]:


BOT_V_BOT_GAMES = load_bot_v_bot_games()


# ## 2.1 - DR100 vs DR30 vs DR10 [DR vs. itself]

# In[9]:


dr_bots = ['Deeprole_100_50', 'Deeprole_30_15', 'Deeprole_10_5']
dr_games = filter_bot_v_bot_games(BOT_V_BOT_GAMES, dr_bots)
dr_payoffs = compute_payoff_matrix(dr_games, dr_bots)


# ### 2.1.1 - Triangle chart

# In[10]:


make_triangle_plot(dr_bots, lambda probs: calc_new_strat(probs, dr_payoffs))


# ### 2.1.2 - Payoff matrix

# In[11]:


dr_payoffs


# ## 2.2 - DR100 vs CFR6K vs ObserveBot [DR vs. hand crafted methods]

# In[30]:


crafted_bots = ['Deeprole', 'ObserveBot', 'CFRBot_6000000']
crafted_games = filter_bot_v_bot_games(BOT_V_BOT_GAMES, crafted_bots)
crafted_payoffs = compute_payoff_matrix(crafted_games, crafted_bots)


# ### 2.2.1 - Triangle Chart

# In[31]:


make_triangle_plot(crafted_bots, lambda probs: calc_new_strat(probs, crafted_payoffs))


# ### 2.2.2 - Payoff Table

# In[32]:


crafted_payoffs


# ## 2.3 - DR100 vs ObserveBot and MOISMCTS [DR vs. existing methods]
# 
# <span style="color: red; font-size: 1.5em;">Can only run 1v1s.</span>

# ### 2.3.1 - DR100 vs. DeepStack (TODO)

# In[ ]:


# Pending running deepstack


# ### 2.3.2 - DR100 vs. MOISMCTS

# In[ ]:


bots = ['Deeprole', 'MOISMCTSBot']
ismcts_games = filter_bot_v_bot_games(BOT_V_BOT_GAMES, bots)
ismcts_payoff = compute_payoff_matrix(ismcts_games, bots)
ismcts_payoff

# This table indicates strict dominance


# # Part 3: Human vs. Bot Comparisons

# In[43]:


HUMAN_V_BOT_GAMES = load_human_v_bot_games()


# ## 3.1 - Winrate Data

# ### 3.1.1 - Overall Winrate

# In[44]:


compare_humans_and_bots(HUMAN_V_BOT_GAMES, [])


# ### 3.1.2 - Breakdown by Role

# #### 3.1.2.1 - By team

# In[45]:


compare_humans_and_bots(HUMAN_V_BOT_GAMES, ['is_resistance'])


# #### 3.1.2.2 - By Merlin

# In[46]:


compare_humans_and_bots(HUMAN_V_BOT_GAMES[HUMAN_V_BOT_GAMES.role == 'Merlin'], [])


# #### 3.1.2.3 - By Assassin

# In[ ]:


compare_humans_and_bots(HUMAN_V_BOT_GAMES[HUMAN_V_BOT_GAMES.role == 'Assassin'], [])


# # Part 4: MOAWinrate Table

# In[69]:


bots = ['Deeprole', 'MOISMCTSBot']
ismcts_games = filter_bot_v_bot_games(BOT_V_BOT_GAMES, bots)
by_player = convert_to_by_player(ismcts_games)
compare_humans_and_bots(by_player, [])


# In[71]:


bots = ['Deeprole', 'ObserveBot']
ismcts_games = filter_bot_v_bot_games(BOT_V_BOT_GAMES, bots)
by_player = convert_to_by_player(ismcts_games)
compare_humans_and_bots(by_player, [])


# In[72]:


bots = ['Deeprole', 'RandomBot']
ismcts_games = filter_bot_v_bot_games(BOT_V_BOT_GAMES, bots)
by_player = convert_to_by_player(ismcts_games)
compare_humans_and_bots(by_player, [])


# In[73]:


bots = ['Deeprole', 'CFRBot_6000000']
ismcts_games = filter_bot_v_bot_games(BOT_V_BOT_GAMES, bots)
by_player = convert_to_by_player(ismcts_games)
compare_humans_and_bots(by_player, [])


# In[105]:


result = create_moawt(BOT_V_BOT_GAMES, HUMAN_V_BOT_GAMES, [
    'ObserveBot', 'RandomBot', 'CFRBot_6000000', 'ISMCTSBot', 'MOISMCTSBot'
])


# In[106]:


result


# In[117]:


prettify_moawt(result)


# In[118]:


print prettify_moawt(result).to_latex()


# In[96]:


df = pd.DataFrame([{'a': 0.0123},{'a': 0.0123},{'a': 0.0123},{'a': 0.0123}])


# In[99]:


df.a.apply(lambda a: '{:0.2f}'.format(a))


# In[ ]:




