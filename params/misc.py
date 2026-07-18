import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
import params.sigma
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

TAU = config['tau']


def compute_tau(r) -> tuple[int, int]:
    # tau: timescale over which position persists
    
    # q becomes an ornstein-uhlenbeck process (mean-reverting)
    # acf(lag) = exp(-lag / theta)
    # with lag 0, acf(0) = 1/e
    # compute characteristic persistence time until acf decays to 1/e

    q = r['q'].dropna().to_numpy()
    f = acf(q, nlags = 900, fft = True)

    return f, np.argmax(f < np.exp(-1))

def compute_half_life(mid, tau = TAU) -> dict:
    # half life: how long it takes for a measurement's weight to half
    # exponentially down-weight older squared increments when computing current vol

    realized_var = ((mid.shift(-tau) - mid)**2) / tau # computing realized variance over tau window

    candidates = ['5min', '10min', '15min', '30min']
    qlikes = {}
    
    for h in candidates:
        ewma_var = params.sigma.estimate_ewma_vol(mid, half_life = h)**2
        df = pd.concat([ewma_var, realized_var], axis = 1, keys = ['ewma_var', 'realized_var'])
        df = params.sigma.slice_window(df).dropna()
        ratio = df['realized_var'] / df['ewma_var'] # realized variance over estimated variance

        n_zero = (ratio == 0).sum()
        print(f'zeros dropped: {n_zero}')
        ratio = ratio[ratio > 0]
        qlikes[h] = (ratio - np.log(ratio) - 1).mean() # QLIKE loss function, penalizes under-forecasting

    return qlikes