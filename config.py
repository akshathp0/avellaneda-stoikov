from pathlib import Path
import yaml
import pandas as pd

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / 'config.yml'
BINANCE_PATH = ROOT / 'binance'
RESULTS_PATH = ROOT / 'results'

with open(CONFIG_PATH) as f:
    raw = yaml.safe_load(f)

def to_seconds(s) -> float:
    if not isinstance(s, str):
        raise TypeError('must be string')
    f = pd.Timedelta(s).total_seconds()
    if f <= 0:
        raise ValueError('must be positive')
    
    return f

K = raw['k']
A = raw['A']
GAMMA = raw['gamma']
QUOTE_SIZE = raw['quote_size']
TAU = to_seconds(raw['tau'])
HALF_LIFE = to_seconds(raw['half_life'])
SIGMA_WINDOW = raw['sigma_window']
GRID_FREQUENCY = raw['grid_frequency']
EWMA_WARMUP = raw['ewma_warmup']
EVENT_GAP = raw['event_gap']
TRAIN_END = raw['train_end']
TEST_START = raw['test_start']
DELTA_MIN = raw['delta_min']
DELTA_MAX = raw['delta_max']
DELTA_POINTS = raw['delta_points']
SLICE_MIN = raw['slice_min'] 
SLICE_MAX = raw['slice_max']