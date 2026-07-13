import numpy as np

class Strategy:
    def __init__(self, gamma, tau, k):
        self.q = 0
        self.cash = 0
        self.gamma = gamma
        self.tau = tau
        self.k = k

    def quote(self, mid, sigma):
        # r(s, t) = s - q * gamma * sigma^2 * tau
        # delta_a + delta_b = gamma * sigma^2 * tau + 2/gamma * ln(1 + gamma / k)

        r = mid - self.q * self.gamma * sigma**2 * self.tau
        spread = self.gamma * sigma**2 * self.tau + (2 / self.gamma) * np.log(1 + self.gamma / self.k)

        self.bid = r - spread / 2
        self.ask = r + spread / 2

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