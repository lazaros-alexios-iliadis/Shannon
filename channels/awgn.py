import torch
from torch import nn, Tensor

class AWGN(nn.Module):
    def __init__(self, snr_db: float | None = None, ebno_db: float | None = None,
                 bits_per_sym: int | None = None, rate: float | None = None):
        super().__init__()
        self.snr_db = snr_db
        self.ebno_db = ebno_db
        self.bits_per_sym = bits_per_sym
        self.rate = rate

    @staticmethod
    def _sigma2_from_snr_db(snr_db: Tensor):
        return (10.0 ** (-snr_db / 10.0)).to(torch.float32)

    @staticmethod
    def _sigma2_from_ebno(ebno_db: Tensor, bits_per_sym: int, rate: float):
        snr_db = ebno_db + 10*torch.log10(torch.tensor(bits_per_sym*rate))
        return AWGN._sigma2_from_snr_db(snr_db)

    def forward(self, x: Tensor, generator: torch.Generator | None = None,
                noise_var: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Add complex AWGN.
        Returns (y, noise_var)
        """
        if noise_var is None:
            if self.snr_db is not None:
                sigma2 = self._sigma2_from_snr_db(torch.tensor(self.snr_db))
            else:
                assert self.ebno_db is not None and self.bits_per_sym and self.rate
                sigma2 = self._sigma2_from_ebno(torch.tensor(self.ebno_db), self.bits_per_sym, self.rate)
        else:
            sigma2 = torch.as_tensor(noise_var, dtype=torch.float32, device=x.device)
        sigma = torch.sqrt(sigma2/2.0)
        noise = torch.randn_like(x.real, generator=generator) * sigma
        noise = noise + 1j * (torch.randn_like(x.real, generator=generator) * sigma)
        y = x + noise.to(x.dtype)
        return y, sigma2
