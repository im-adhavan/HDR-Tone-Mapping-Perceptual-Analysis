import torch
from .gpu_utils import luminance


def dynamic_range(hdr):
    Y = luminance(hdr)
    Y = Y[Y > 0]
    p99 = torch.quantile(Y, 0.999)
    p01 = torch.quantile(Y, 0.001)
    return torch.log10(p99 / p01).item()


def log_luminance_std(hdr):
    Y = torch.log10(1e-6 + luminance(hdr))
    return torch.std(Y).item()


def highlight_energy_ratio(hdr):
    Y = luminance(hdr)
    threshold = torch.quantile(Y, 0.99)
    return (torch.sum(Y[Y >= threshold]) / torch.sum(Y)).item()


def shadow_mass_ratio(hdr):
    Y = luminance(hdr)
    threshold = torch.quantile(Y, 0.01)
    return (torch.sum(Y[Y <= threshold]) / torch.sum(Y)).item()


def skewness(hdr):
    Y = luminance(hdr)
    mean = torch.mean(Y)
    std = torch.std(Y)
    return torch.mean(((Y - mean) / (std + 1e-6))**3).item()


def kurtosis(hdr):
    Y = luminance(hdr)
    mean = torch.mean(Y)
    std = torch.std(Y)
    return torch.mean(((Y - mean) / (std + 1e-6))**4).item()