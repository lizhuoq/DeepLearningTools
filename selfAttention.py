import doctest

import torch

from package import *


def selfAttention(current_state: torch.Tensor, states: List):
    '''
    >>> cs = torch.randn(2, 4, 8)
    >>> states = [cs]
    >>> assert (cs == selfAttention(cs, states)).sum() == cs.reshape(-1).__len__()

    >>> states = [torch.randn(2, 4, 8) for _ in range(6)] + [cs]
    >>> selfAttention(cs, states).shape
    torch.Size([2, 4, 8])
    '''
    # state shape: D * num_layers, batch_size, num_hiddens
    scores = []
    for state in states:
        score = (current_state * state).sum(dim=-1)
        scores.append(score)
    scores = torch.stack(scores, dim=-1)
    alphas = F.softmax(scores, dim=-1)

    length = len(states)
    res = []
    for i in range(length):
        res.append(states[i] * alphas[:, :, i].unsqueeze(-1))
    return torch.stack(res, dim=-1).sum(dim=-1)


if __name__ == '__main__':
    doctest.testmod()