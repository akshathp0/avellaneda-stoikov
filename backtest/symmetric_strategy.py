class SymmetricStrategy:
    def __init__(self, half_spread):
        self.q = 0
        self.cash = 0
        self.half_spread = half_spread
    
    def quote(self, mid, sigma) -> tuple[float, float]:
        self.bid = mid - self.half_spread
        self.ask = mid + self.half_spread

        return self.bid, self.ask
    
    def fill(self, side, price, quantity):
        if side == 'bid':
            self.q += quantity
            self.cash -= price * quantity
        elif side == 'ask':
            self.q -= quantity
            self.cash += price * quantity
        else:
            raise ValueError('side must be bid or ask')