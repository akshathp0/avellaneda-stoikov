class Backtester:
    # at each step t: find mid[t], sigma[t] info up to t
    # calculate quotes for interval (t, t + dt]
    # check if events in interval (t, t + dt] fill quotes
    # fills mutate strategy state
    # half-open interval -> events at t predate quote for accuracy

    