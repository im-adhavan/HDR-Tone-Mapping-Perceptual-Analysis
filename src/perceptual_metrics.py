import torch
from .gpu_utils import luminance


def clipping_ratio(img):
    mask = ((img <= 0) | (img >= 1)).float()
    return torch.mean(mask).item()


def contrast_loss(hdr, ldr):
    Y_hdr = torch.log10(1e-6 + luminance(hdr))
    Y_ldr = torch.log10(1e-6 + luminance(ldr))
    return (torch.std(Y_hdr) - torch.std(Y_ldr)).item()


def entropy(img):
    Y = luminance(img)
    hist = torch.histc(Y, bins=256, min=0.0, max=1.0)
    hist = hist / torch.sum(hist)
    hist += 1e-12
    return (-torch.sum(hist * torch.log2(hist))).item()