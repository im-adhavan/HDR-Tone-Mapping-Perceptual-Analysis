import torch
import torch.nn.functional as F
from .gpu_utils import luminance


class ToneMapper:

    def __init__(self, operator_name, **params):
        self.operator_name = operator_name
        self.params = params

    def apply(self, hdr):
        if self.operator_name == "reinhard":
            return self._reinhard(hdr)
        elif self.operator_name == "drago":
            return self._drago(hdr)
        elif self.operator_name == "filmic":
            return self._filmic(hdr)
        elif self.operator_name == "gamma":
            return self._gamma(hdr)
        elif self.operator_name == "reinhard_local":
            return self._reinhard_local(hdr)
        else:
            raise ValueError("Unknown operator")


    def _reinhard(self, hdr):
        a = self.params.get("exposure", 1.0)
        white = self.params.get("white", None)

        L = luminance(hdr)
        L_scaled = a * L

        if white is not None:
            L_mapped = (L_scaled * (1 + L_scaled / (white**2))) / (1 + L_scaled)
        else:
            L_mapped = L_scaled / (1 + L_scaled)

        return torch.clamp(hdr * (L_mapped / (L + 1e-6)).unsqueeze(-1), 0, 1)


    def _drago(self, hdr):
        bias = self.params.get("bias", 0.85)

        L = luminance(hdr)
        Lmax = torch.max(L)
        Ld = torch.log1p(bias * L) / torch.log1p(Lmax + 1e-6)

        return torch.clamp(hdr * (Ld / (L + 1e-6)).unsqueeze(-1), 0, 1)


    def _filmic(self, hdr):
        A, B, C, D, E, Fp = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
        x = hdr
        mapped = ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*Fp)) - E/Fp
        return torch.clamp(mapped, 0, 1)

    def _gamma(self, hdr):
        gamma = self.params.get("gamma", 2.2)
        return torch.clamp(hdr ** (1/gamma), 0, 1)


    def _reinhard_local(self, hdr):
        sigma = self.params.get("sigma", 3.0)

        L = luminance(hdr).unsqueeze(0).unsqueeze(0)
        kernel_size = int(2 * sigma + 1)

        blur = F.avg_pool2d(L, kernel_size, stride=1, padding=kernel_size//2)
        blur = blur.squeeze()

        L_scaled = L.squeeze() / (1 + blur)
        return torch.clamp(hdr * (L_scaled / (luminance(hdr) + 1e-6)).unsqueeze(-1), 0, 1)