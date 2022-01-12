import torch 
import torch.nn as nn

# criteria for classifier with labeled and unlabeled data
clf_loss = nn.NLLLoss()

# criteria for classifier with fake data
def inverted_cross_entropy(y_pred, y_true):
    out = - torch.mean(y_true * torch.log(1-y_pred + 1e-6) + 1e-6)
    return out

# criterion for discriminator and generator
bce_loss = nn.BCEWithLogitsLoss()

def d_loss(m, s, real, fake, y):
    real = real - m
    return bce_loss(s*(real - fake) + 1e-6 , y)

def g_loss(m, s, real, fake, y):
    fake = fake + m
    return bce_loss(s*(fake - real) + 1e-6, y)