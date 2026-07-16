import numpy as np
import pandas as pd

class Backtester:
    # at each step t: find mid[t], sigma[t] info up to t
    # calculate quotes for interval (t, t + dt]
    # check if events in interval (t, t + dt] fill quotes
    # fills mutate strategy state
    # half-open interval -> events at t predate quote for accuracy

    def __init__(self, strategy, size):
        self.strategy = strategy
        self.size = size

    def loop(self, mid, sigma, events) -> pd.DataFrame:
        # dataframes to np array for speed

        mids = mid['mid_price'].to_numpy()
        grid_times = mid.index.to_numpy()
        sigmas = sigma.to_numpy()
        ev_times = events.index.to_numpy()
        ev_flags = events['is_buyer_maker'].to_numpy()
        ev_pmin = events['price_min'].to_numpy()
        ev_pmax = events['price_max'].to_numpy()
        ev_id = events['event_id'].to_numpy()

        n_steps = len(grid_times) - 1

        # diagnostic arrays
        bid_filled = np.zeros(n_steps, dtype = bool) # filled or not
        ask_filled = np.zeros(n_steps, dtype = bool)

        bid_quote = np.full(n_steps, np.nan) # price
        ask_quote = np.full(n_steps, np.nan)

        q = np.full(n_steps, np.nan) # wealth stats
        cash = np.full(n_steps, np.nan)
        mid_d = np.full(n_steps, np.nan)
        wealth = np.full(n_steps, np.nan)

        bid_idx = np.full(n_steps, np.nan)
        ask_idx = np.full(n_steps, np.nan)

        boundaries = np.searchsorted(ev_times, grid_times, side = 'right') # count events <= timestamp t

        for t in range(n_steps):
            ev_slice = slice(boundaries[t], boundaries[t+1]) # this step's events

            if not np.isnan(sigmas[t]): # skip over NaNs from half-life warm-up
                bid, ask = self.strategy.quote(mids[t], sigmas[t])

                bid_check = ev_flags[ev_slice] & (ev_pmin[ev_slice] <= bid)
                ask_check = ~ev_flags[ev_slice] & (ev_pmax[ev_slice] >= ask)

                if bid_check.any():
                    bid_filled[t] = True
                    bid_idx[t] = ev_id[boundaries[t] + np.argmax(bid_check)] # boundary start + fill index mapped to id
                    self.strategy.fill('bid', bid, self.size)
                
                bid_quote[t] = bid

                if ask_check.any():
                    ask_filled[t] = True
                    ask_idx[t] = ev_id[boundaries[t] + np.argmax(ask_check)]
                    self.strategy.fill('ask', ask, self.size)
                
                ask_quote[t] = ask
        
            q[t] = self.strategy.q
            cash[t] = self.strategy.cash
            wealth[t] = cash[t] + q[t] * mids[t]
            mid_d[t] = mids[t]
            
        return pd.DataFrame({'bid_filled': bid_filled, 'ask_filled': ask_filled,
                             'bid_quote': bid_quote, 'ask_quote': ask_quote,
                             'bid_id': bid_idx, 'ask_id': ask_idx,
                             'q': q, 'cash': cash, 'wealth': wealth, 'mid': mid_d},
                             index = grid_times[:-1])