import pandas as pd
import numpy as np
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

EVENT_GAP = config['event_gap']
DELTA_MIN = config['delta_min']
DELTA_MAX = config['delta_max']
DELTA_POINTS = config['delta_points']
SLICE_MIN = config['slice_min']
SLICE_MAX = config['slice_max']

def match_trades(at, deduped) -> pd.DataFrame:
    at_c = at.copy()
    at_c = at_c.sort_values('agg_trade_id')
    at_c['transact_time'] = pd.to_datetime(at_c['transact_time'], unit = 'ms')
    
    if not at_c['transact_time'].is_monotonic_increasing:
        raise ValueError('time not monotonically increasing')

    matched = pd.merge_asof(at_c[['first_trade_id', 'last_trade_id', 'price', 'quantity', 'transact_time', 'is_buyer_maker']], 
                            deduped[['mid_price']],
                            left_on = 'transact_time',
                            right_index = True,
                            direction = 'backward',
                            allow_exact_matches = False) # match trades with mid price at time of trade
    
    matched = matched.set_index('transact_time')

    return matched

def calculate_spread(matched) -> pd.DataFrame:
    # is_buyer_maker = True -> seller matched bid (sells below mid, buyer wants cheap)
    # spread: mid_price - price

    spreads = matched.copy()
    spreads['depth'] = np.where(spreads['is_buyer_maker'],
                                spreads['mid_price'] - spreads['price'],
                                spreads['price'] - spreads['mid_price'])

    return spreads

def event_aggregation(spreads, event_gap = EVENT_GAP):
     # check whether first trade ID is not consecutive with previous last trade ID
     # or aggressor changed 
     # or time between trades changed by over 1ms

    events = spreads.copy()
    events['new_event'] = ((events['first_trade_id'] != events['last_trade_id'].shift(1) + 1) | 
                            (events['is_buyer_maker'] != events['is_buyer_maker'].shift(1)) |
                            (events.index.diff() > pd.Timedelta(event_gap))) 
    events['event_id'] = events['new_event'].cumsum()
    events = events.reset_index() # change transact_time from index to normal column for aggregation

    events_agg = events.groupby('event_id').agg(
        max_depth = ('depth', 'max'),
        trade_quantity = ('quantity', 'sum'),
        is_buyer_maker = ('is_buyer_maker', 'first'),
        transact_time = ('transact_time', 'first'),
        price_min = ('price', 'min'),
        price_max = ('price', 'max'),
    )

    events_agg['event_id'] = events_agg.index
    events_agg = events_agg.set_index('transact_time')

    return events_agg

def delta_grid(min = DELTA_MIN, max = DELTA_MAX, points = DELTA_POINTS) -> np.ndarray:
    return np.geomspace(min, max, points) # geometric progression of delta values

def survival_counts(events) -> pd.DataFrame:
    # go thru delta grid and count how many bids / asks
    # would be filled by max_depth
    # on sliced train data

    grid = delta_grid()

    is_bid = events['is_buyer_maker'] # mask as series
    sorted_true = events.loc[is_bid, 'max_depth'].sort_values().to_numpy()
    sorted_false = events.loc[~is_bid, 'max_depth'].sort_values().to_numpy()

    counts = pd.DataFrame(index = grid)
    counts['bid_fills'] = len(sorted_true) - np.searchsorted(sorted_true, grid, side = 'left')
    counts['ask_fills'] = len(sorted_false) - np.searchsorted(sorted_false, grid, side = 'left')

    delta_t = (events.index[-1] - events.index[0]).total_seconds()
    counts['lambda_bid'] = counts['bid_fills'] / delta_t
    counts['lambda_ask'] = counts['ask_fills'] / delta_t

    return counts

def slice_delta(counts, min = SLICE_MIN, max = SLICE_MAX) -> pd.DataFrame:
    return counts.loc[min:max]

def regress_intensity(x, y) -> tuple[int, int]:
    # A: base arrival rate (fills/sec at the mid)
    # k: quote depth decay rate (1/dollars)

    # lambda(delta) = Ae^(-k*delta)
    # ln(lambda) = ln(A) - k*delta
    # slope is -k, intercept is ln(A)

    y = np.log(y)
    k_neg, log_A = np.polyfit(x, y, 1)
    k = -k_neg
    A = np.exp(log_A)

    return A, k