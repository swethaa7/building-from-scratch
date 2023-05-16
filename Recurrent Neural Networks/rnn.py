import numpy as np
from numpy.random import randn

class RNN:
    def __init__(self, input_size, output_size, hidden_size = 64):

        self.whh = randn(hidden_size, hidden_size)/1000
        self.wxh = randn(hidden_size, input_size)/1000
        self.why = randn(output_size, hidden_size)/1000
        #dividing by 1000 to reduce the initial variance of our weights.

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = {0: h}

        for i, x in enumerate(inputs):
            h = np.tanh(self.wxh @ x + self.whh @ h + self.bh)
            self.last_hs[i+1] = h

        y = self.why @ h + self.by

        return y, h
    
    def backprop(self, dy, learn_rate = 2e-2):
        n = len(self.last_inputs)

        d_why = dy @ self.last_hs[n].T
        d_by = dy

        d_whh = np.zeros(self.whh.shape)
        d_wxh = np.zeros(self.wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        d_h = self.why.T @ dy
        for t in reversed(range(n)):
            # An intermediate value: dL/dh * (1 - h^2)
            temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)

            # dL/db = dL/dh * (1 - h^2)
            d_bh += temp

            # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
            d_whh += temp @ self.last_hs[t].T

            # dL/dWxh = dL/dh * (1 - h^2) * x
            d_wxh += temp @ self.last_inputs[t].T

            # Next dL/dh = dL/dh * (1 - h^2) * Whh
            d_h = self.whh @ temp

    # Clip to prevent exploding gradients.
        for d in [d_wxh, d_whh, d_why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)   # to clip a limit value in an input array

    # Update weights and biases using gradient descent.
        self.whh -= learn_rate * d_whh
        self.wxh -= learn_rate * d_wxh
        self.why -= learn_rate * d_why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by


