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

def run(mid, sigma, events,
        strategy_cls = backtest.strategy.Strategy, strategy_kwargs = None,
        size = QUOTE_SIZE, start = None, end = TRAIN_END) -> pd.DataFrame:
    
    mid = params.sigma.slice_window(mid, start = start, end = end)
    sigma = params.sigma.slice_window(sigma, start = start, end = end)
    events = params.sigma.slice_window(events, start = start, end = end)

    kwargs = strategy_kwargs or {'gamma': GAMMA, 'tau': TAU, 'k': K}
    strategy = strategy_cls(**kwargs)
    backtester = backtest.backtester.Backtester(strategy, size)

    return backtester.loop(mid, sigma, events)

def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bt, at = data.d_load.load()
    mid, deduped = params.sigma.build_mid(bt)
    sigma = params.sigma.estimate_ewma_vol(mid)

    matched = params.intensity.match_trades(at, deduped)
    spreads = params.intensity.calculate_spread(matched)
    events = params.intensity.event_aggregation(spreads)

    return mid, sigma, events