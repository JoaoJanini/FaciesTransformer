from torch import nn
import torch

# For bidirectional LSTMs, h_n is not equivalent to the last element of output; the former contains the final forward and reverse hidden states, while the latter contains the final forward hidden state and the initial reverse hidden state.
rnn = nn.LSTM(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))
