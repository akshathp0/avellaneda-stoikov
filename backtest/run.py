import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import params.intensity
import params.sigma
import backtest.strategy
import backtest.backtester
import data.d_load

import pandas as pd
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

K = config['k']
GAMMA = config['gamma']
TAU = config['tau']
QUOTE_SIZE = config['quote_size']

TRAIN_END = config['train_end']

def run(k = K, gamma = GAMMA, tau = TAU, size = QUOTE_SIZE, start = None, end = TRAIN_END) -> pd.DataFrame:
    mid, sigma, events = load_artifacts()
    
    mid = params.sigma.slice_window(mid, start = start, end = end)
    sigma = params.sigma.slice_window(sigma, start = start, end = end)
    events = params.sigma.slice_window(events, start = start, end = end)

    strategy = backtest.strategy.Strategy(gamma = gamma, tau = tau, k = k)
    backtester = backtest.backtester.Backtester(strategy, size)

    results = backtester.loop(mid, sigma, events)

    return results

def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bt, at = data.d_load.load()
    mid, deduped = params.sigma.build_mid(bt)
    sigma = params.sigma.estimate_ewma_vol(mid)

    matched = params.intensity.match_trades(at, deduped)
    spreads = params.intensity.calculate_spread(matched)
    events = params.intensity.event_aggregation(spreads)

    return mid, sigma, events


