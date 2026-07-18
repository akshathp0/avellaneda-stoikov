import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from statsmodels.tsa.stattools import acf
import params.sigma
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

TAU = config['tau']


def compute_tau(r):
    # tau: timescale over which position persists
    
    # q becomes an ornstein-uhlenbeck process (mean-reverting)
    # acf(lag) = exp(-lag / theta)
    # with lag 0, acf(0) = 1/e
    # compute characteristic persistence time until acf decays to 1/e

    q = r['q'].dropna().to_numpy()
    f = acf(q, nlags = 900, fft = True)

    return f, np.argmax(f < np.exp(-1))

def compute_half_life(mid, tau = TAU):
    # half life: how long it takes for a measurement's weight to half
    # exponentially down-weight older squared increments when computing current vol

    realized_var = ((mid.shift(-tau) - mid)**2) / tau # computing realized variance over tau window

    candidates = ['5min', '10min', '15min', '30min']
    scores = {}
    
    for h in candidates:
        ewma_var = params.sigma.estimate_ewma_vol(mid)
        