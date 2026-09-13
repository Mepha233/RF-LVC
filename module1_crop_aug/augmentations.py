"""Module 1 - data augmentation (RF-LVC).

Perturbation applied to the second augmented view, as described in
"Data Augmentation and Encoding".  The original project file also
defined `scaling` and `permutation`; they are never called by the
training pipeline and were dropped.
"""

import torch


def DataTransform(sample, scaling_rate):
    return jitter(sample, scaling_rate)


def jitter(x, sigma=0.8):
    x_cpu = x.detach().cpu().to(torch.float32)
    noise = torch.randn_like(x_cpu) * float(sigma)
    return x_cpu + noise
