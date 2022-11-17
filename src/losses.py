import torch
import torch.nn as nn

def cos_similarity(a, b):
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b))

class ContrastiveLoss(nn.Module):
    """
    :param sim: a similarity function. it is assumed that if a is more similar to b than c,
    sim(a, b) > sim(a, c)
    :param lamb: the lambda in our formulation
    :param c: the c in our formulation
    """
    def __init__(self, sim, lamb=0.5, c=-0.1):
        super(ContrastiveLoss, self).__init__()
        self.sim = sim
        self.lamb = lamb
        self.c = c

    def forward(self, c0, i0, c1, i1):
        caption_sim = self.sim(c0, c1)
        pair0_sim = self.sim(c0, i0)
        pair1_sim = self.sim(c1, i1)
        score = caption_sim - self.lamb * (pair0_sim + pair1_sim) + self.c
        zero = torch.tensor(0)

        return torch.maximum(zero, score)

# testing code
if __name__ == "__main__":
    loss = ContrastiveLoss(cos_similarity)
    c0 = torch.tensor([1., 1.], dtype=torch.float32)
    i0 = torch.tensor([-1., 2.], dtype=torch.float32)
    c1 = torch.tensor([1.1, 1.], dtype=torch.float32)
    i1 = torch.tensor([1.2, 1.1], dtype=torch.float32)

    # c0 c1 should be close, c0 i0 should be not close, c0 i1 should be close
    print(cos_similarity(c0, c1))
    print(cos_similarity(c0, i0))
    print(cos_similarity(c0, i1))

    high = loss(c0, i0, c1, i1)
    low = loss(i0, c0, i1, c1)

    # high should be higher than low
    print(high, low)
