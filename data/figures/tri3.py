import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Memory
memory = Memory('/tmp/joblib', verbose=1)

import egtsimplex
from egtplot import plot_static

from scipy.special import comb

def scipy_multinomial(params):
    if len(params) == 1:
        return 1
    coeff = (comb(np.sum(params), params[-1], exact=True) *
             scipy_multinomial(params[:-1]))
    return coeff

def P(N_i, x):
    x = np.array(x)
    N_i = np.array(N_i)
    return scipy_multinomial(N_i) * np.prod( x ** N_i )

def r(x, table):
    x = np.array(x)
    numerator = np.zeros(len(x))
    for index, payoff in table.iterrows():
        numerator += P(index, x) * np.array(payoff)
    denominator = 1.0 - (1.0 - x) ** 5
    print(numerator / denominator)
    return numerator / denominator

def calc_xdot(x, table):
    rx = r(x, table)
    xtAx = np.sum(x * rx)
    xdot = x * (rx - xtAx)
    return xdot

LEARNING_RATE = 1
def calc_new_strat(x, table):
    return LEARNING_RATE*calc_xdot(x, table)

payoffs = pd.read_pickle('dr.pik')

def f(x,t):
    A=np.array([[0,1,-1],[-1,0,1],[1,-1,0]])
    phi=(x.dot(A.dot(x)))
    x0dot=x[0]*(A.dot(x)[0]-phi)
    x1dot=x[1]*(A.dot(x)[1]-phi)
    x2dot=x[2]*(A.dot(x)[2]-phi)
    return [x0dot,x1dot,x2dot]

dynamics=egtsimplex.simplex_dynamics(lambda probs, t: calc_new_strat(probs, payoffs))    
# dynamics=egtsimplex.simplex_dynamics(f)    
import pdb; pdb.set_trace()

#initialize simplex_dynamics object with function
# dynamics=egtsimplex.simplex_dynamics(f)

#plot the simplex dynamics
fig,ax=plt.subplots()
dynamics.plot_simplex(ax, typelabels=['DR100', 'DR30', 'DR10'])
plt.show()
