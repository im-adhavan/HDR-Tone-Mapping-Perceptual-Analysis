import torch


def rgb_to_uv(img):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]

    denom = R + G + B + 1e-6
    u = R / denom
    v = G / denom
    return u, v


def chromaticity_shift(hdr, ldr):
    u1, v1 = rgb_to_uv(hdr)
    u2, v2 = rgb_to_uv(ldr)

    shift = torch.sqrt((u1 - u2)**2 + (v1 - v2)**2)
    return torch.mean(shift).item()