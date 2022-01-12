import torch 
import torch.nn as nn
import math 
from scipy.special import binom


# criteria for classifier with labeled and unlabeled data
clf_loss = nn.NLLLoss()

# criteria for classifier with fake data
def inverted_cross_entropy(y_pred, y_true):
    out = - torch.mean(y_true * torch.log(1-y_pred + 1e-6) + 1e-6)
    return out

# criterion for discriminator and generator
bce_loss = nn.BCEWithLogitsLoss()

def calculate_phi_theta(cos_theta, device, m=4): # m: margin
    C_m_2n = torch.Tensor(binom(m, range(0, m + 1, 2))).to(device)  # C_m^{2n}
    cos_powers = torch.Tensor(range(m, -1, -2)).to(device)  # m - 2n
    sin2_powers = torch.Tensor(range(len(cos_powers))).to(device)  # n
    signs = torch.ones(m // 2 + 1).to(device)  # 1, -1, 1, -1, ...
    signs[1::2] = -1

    sin2_theta = 1 - cos_theta**2
    cos_terms = cos_theta ** cos_powers # cos^{m - 2n}
    sin2_terms = sin2_theta ** sin2_powers # sin2^{n}

    cos_m_theta = (signs *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                    C_m_2n *
                    cos_terms *
                    sin2_terms).sum(1)  # summation of all terms # shape: batch_size

    k = find_k(cos_m_theta, m)
    phi_theta = (-1) ** k * cos_m_theta - 2 * k
    return phi_theta.unsqueeze(1)

def find_k(cos_theta, m):
    divisor = math.pi / m  # pi/m
    # to account for acos numerical errors
    eps = 1e-7
    cos_theta = torch.clamp(cos_theta, -1 + eps, 1 - eps)
    theta = cos_theta.acos() # arccos
    k = (theta / divisor).floor().detach()
    return k

def d_loss(real, fake, y, device, m=4):
    real = calculate_phi_theta(real, device, m)
    return bce_loss(real - torch.mean(fake) + 1e-6, y)

def g_loss(real, fake, y, device, m=4):
    fake = calculate_phi_theta(fake, device, m)
    return bce_loss(fake - torch.mean(real) + 1e-6, y)
