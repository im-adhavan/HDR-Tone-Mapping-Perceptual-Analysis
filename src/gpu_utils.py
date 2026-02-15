import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor(np_img):
    return torch.from_numpy(np_img).to(DEVICE)


def luminance(img):
    return 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]