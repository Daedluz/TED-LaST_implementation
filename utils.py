import torch
from torch import nn

import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.special import binom
# --------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# Probability and Log-Odds Functions
# Probably not needed for the current task, but useful for further experiments
# --------------------------------------------------------------------------------
def collect_probability_differences(model, data_loader, device):
    """
    Collects absolute probability differences: abs(2P - 1) for each sample in data_loader.
    """
    probability_differences = []
    model.eval()

    with torch.no_grad():
        for item, _ in tqdm(data_loader):
            item = item.to(device)

            predictions = model(item)
            prob_diff = torch.abs(2 * predictions - 1)
            probability_differences.extend(prob_diff.cpu().numpy())

    return np.array(probability_differences)

def calculate_log_odds(prob_array):
    """
    Calculates log-odds for a given array of probabilities.
    Map probabilities from (0, 1) to (-inf, +inf).
    """
    epsilon = 1e-5
    prob_array = np.clip(prob_array, epsilon, 1 - epsilon)
    return np.log(prob_array / (1 - prob_array))

def calculate_log_odds_sum(input_array):
    """
    Transforms input from (0, +inf) to (-inf, +inf) using the natural logarithm.
    """
    epsilon = 1e-5  # Small constant to avoid issues with very small values
    input_array = np.clip(input_array, epsilon, None)  # Clip to avoid zero or negative inputs
    return np.log(input_array)

def fit_normal_distribution(data_array):
    """
    Returns the mean and variance of the data_array.
    """
    mean = np.mean(data_array)
    variance = np.var(data_array)
    return mean, variance

def save_distribution_results(filename, in_mean, in_variance):
    df = pd.DataFrame({
        'in_mean': [in_mean],
        'in_variance': [in_variance],
        # 'out_mean': [out_mean],
        # 'out_variance': [out_variance]
    })
    df.to_csv(filename, index=False)
    print(f"Saved distribution data to {filename}")

# --------------------------------------------------------------------------------
# L-Softmax Linear Layer code copied from adaptive blend original code
# --------------------------------------------------------------------------------
class LSoftmaxLinear(nn.Module):

    def __init__(self, input_features, output_features, margin):
        super().__init__()
        self.input_dim = input_features  # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.margin = margin  # m
        self.beta = 100
        self.beta_min = 0
        self.scale = 0.99


        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).cuda()  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).cuda()  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).cuda()  # n
        self.signs = torch.ones(margin // 2 + 1).cuda()  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta**2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            x, w = input, self.weight
            beta = max(self.beta, self.beta_min)

            logit = x.mm(w)
            #logit = torch.matmul(x,w)
            indexes = range(logit.size(0))
            logit_target = logit[indexes, target]

            # cos(theta) = w * x / ||w||*||x||
            w_target_norm = w[:, target].norm(p=2, dim=0)
            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            # equation 7
            cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

            # find k in equation 6
            k = self.find_k(cos_theta_target)

            # f_y_i
            logit_target_updated = (w_target_norm *
                                    x_norm *
                                    (((-1) ** k * cos_m_theta_target) - 2 * k))
            logit_target_updated_beta = (logit_target_updated + beta * logit[indexes, target]) / (1 + beta)

            logit[indexes, target] = logit_target_updated_beta
            self.beta *= self.scale
            return logit
        else:
            assert target is None
            return input.mm(self.weight)
