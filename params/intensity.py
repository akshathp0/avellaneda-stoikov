import pandas as pd
import numpy as np
import yaml

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

EVENT_GAP = config['event_gap']

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
    # is_buyer_maker = True -> seller is aggressor (sells below mid)
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
        transact_time = ('transact_time', 'first')
    )

    return events_agg