from dataclasses import dataclass
import torch
from torch import Tensor

@dataclass
class QAM:
    M: int = 4
    gray: bool = True
    normalize: bool = True
    dtype: torch.dtype = torch.complex64
    device: torch.device | None = None

    def __post_init__(self):
        m = int(torch.log2(torch.tensor(self.M)).item())
        assert 2**m == self.M, "M must be power of two"
        # Build 1D PAM points then cartesian product for square QAM
        k = int((m)//2)
        pam = torch.arange(-(2**k - 1), (2**k), 2, dtype=torch.float32)
        # Gray encode indices
        if self.gray:
            idx = torch.arange(self.M)
            gray_idx = idx ^ (idx >> 1)
        else:
            gray_idx = torch.arange(self.M)
        # 2D mapping (row-major)
        x = pam.repeat_interleave(2**k)
        y = pam.repeat(2**k)
        const = x + 1j * y
        if self.normalize:
            const = const / const.abs().pow(2).mean().sqrt()
        # Reorder by gray index
        self.symbols = const[gray_idx].to(self.dtype)
        self.bits = gray_idx.unsqueeze(-1).bitwise_right_shift(
            torch.arange(m - 1, -1, -1)).bitwise_and(1).to(torch.int8)
        if self.device:
            self.symbols = self.symbols.to(self.device)
            self.bits = self.bits.to(self.device)

    @property
    def bits_per_sym(self):
        return int(torch.log2(torch.tensor(self.M)))

    def llr(self, y: Tensor, noise_var: Tensor, max_log: bool = True) -> Tensor:
        """Compute LLRs per bit for received symbols y.
        y: [..., N]
        noise_var: broadcastable to y (real scalar)
        return: [..., N, m]
        """
        m = self.bits_per_sym
        const = self.symbols.view(1, -1)
        y = y.view(*y.shape, 1)
        d2 = (y - const).abs().pow(2).real  # [..., N, M]
        llrs = []
        for b in range(m):
            mask0 = (self.bits[:, b] == 0)
            mask1 = ~mask0
            d0 = d2[..., mask0]
            d1 = d2[..., mask1]
            if max_log:
                s0 = (-d0 / noise_var).amin(dim=-1)
                s1 = (-d1 / noise_var).amin(dim=-1)
            else:
                s0 = torch.logsumexp(-d0 / noise_var, dim=-1)
                s1 = torch.logsumexp(-d1 / noise_var, dim=-1)
            llrs.append(s1 - s0)
        return torch.stack(llrs, dim=-1)
