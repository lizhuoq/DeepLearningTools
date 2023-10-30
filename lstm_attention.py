import torch

from package import *


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, dropout=dropout, batch_first=True)

    def forward(self, x):
        """

        :param x: shape: (batch_size, seq_len, input_size)
        :return: output, (h_n, c_n)
        output shape: batch_size, seq_len, D * hidden_size
        h_n shape: D * num_layers, batch_size, hidden_size
        c_n shape: D * num_layers, batch_size, hidden_size
        D = 1 if bidirectional == False else 2
        """
        return self.lstm(x)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, queries, keys, values):
        """

        :param queries: shape: (batch_size, n_of_queries, d_qk)
        :param keys: shape: (batch_size, n_of_kv, d_qk)
        :param values: shape: (batch_size, n_of_kv, d_v)
        :return: shape: (batch_size, n_of_queries, d_v)
        """
        scores = torch.bmm(queries, keys.transpose(1, 2))
        alphas = F.softmax(scores, dim=-1)
        return torch.bmm(alphas, values)


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.attention = Attention()
        self.lstmcell = nn.LSTMCell(input_size=input_size, hidden_size=hidden_size)

    def forward(self, x, h_0, c_0, enc_states):
        batch_size, seq_len, input_size = x.shape
        res = []
        for i in range(seq_len):
            h_c_0 = self.attention(h_0.unsqueeze(1), enc_states, enc_states).squeeze(1)
            c_c_0 = self.attention(c_0.unsqueeze(1), enc_states, enc_states).squeeze(1)
            input = torch.concat([x[:, i, :], h_c_0, c_c_0], dim=1)
            h_1, c_1 = self.lstmcell(input, (h_0, c_0))
            res.append(h_1)
            h_0, c_0 = h_1, c_1

        return torch.stack(res, dim=1)


class Model(nn.Module):
    """
    >>> model = Model(8, 16, 2, True, 0, 1)
    >>> x = torch.randn(2, 16, 8)
    >>> model(x).shape
    torch.Size([2, 16, 1])

    >>> model = Model(8, 16, 1, False, 0, 1)
    >>> model(x).shape
    torch.Size([2, 16, 1])
    """
    def __init__(self, enc_input_size, enc_hidden_size, num_layers, bidirectional, dropout, dec_out):
        super().__init__()
        self.encoder = Encoder(enc_input_size, enc_hidden_size, num_layers, bidirectional, dropout)
        if bidirectional:
            dec_input_size = enc_input_size + 2 * 2 * enc_hidden_size
            dec_hidden_size = 2 * enc_hidden_size
        else:
            dec_input_size = enc_input_size + 2 * enc_hidden_size
            dec_hidden_size = enc_hidden_size
        self.decoder = Decoder(dec_input_size, dec_hidden_size)
        self.projection = nn.Linear(dec_hidden_size, dec_out)
        self.bidirectional = bidirectional

    def forward(self, x):
        """

        :param x: shape: batch_size, seq_len, input_size
        :return: shape: batch_size, seq_len, dec_out
        """
        D = 2 if self.bidirectional else 1
        enc_output, (h_n, c_n) = self.encoder(x)
        h_n, c_n = h_n[-D:], c_n[-D:]
        h_n = torch.concat([h_n[i] for i in range(D)], dim=1)
        c_n = torch.concat([c_n[i] for i in range(D)], dim=1)
        dec_output = self.decoder(x, h_n, c_n, enc_output)
        return self.projection(dec_output)


if __name__ == '__main__':
    doctest.testmod()
