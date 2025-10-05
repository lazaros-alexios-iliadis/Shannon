import torch


def db2lin(x_db):
    """
    10^(x_db/10). Accepts float or Tensor. Returns Tensor (float32).
    """
    return torch.as_tensor(x_db, dtype=torch.float32).div(10.0).pow(10.0)


def lin2db(x_lin):
    """
    10*log10(x_lin). Accepts float or Tensor. Returns Tensor (float32).
    """
    x_lin = torch.as_tensor(x_lin, dtype=torch.float32)
    if torch.any(x_lin <= 0):
        raise ValueError("lin2db expects strictly positive inputs.")
    return 10.0 * torch.log10(x_lin)


# -------- Spectral efficiency helpers --------
# spectral_efficiency = Rb/B = code_rate * bits_per_symbol (when Es is per symbol)

def snr_to_ebn0(snr_db, spectral_efficiency):
    """
    Convert SNR [dB] -> Eb/N0 [dB].
    Eb/N0 = SNR - 10*log10(Rb/B)
    """
    snr_db = torch.as_tensor(snr_db, dtype=torch.float32)
    se = torch.as_tensor(spectral_efficiency, dtype=torch.float32)
    if torch.any(se <= 0):
        raise ValueError("spectral_efficiency must be > 0")
    return snr_db - 10.0 * torch.log10(se)


def ebn0_to_snr(ebn0_db, spectral_efficiency):
    """
    Convert Eb/N0 [dB] -> SNR [dB].
    SNR = Eb/N0 + 10*log10(Rb/B)
    """
    ebn0_db = torch.as_tensor(ebn0_db, dtype=torch.float32)
    se = torch.as_tensor(spectral_efficiency, dtype=torch.float32)
    if torch.any(se <= 0):
        raise ValueError("spectral_efficiency must be > 0")
    return ebn0_db + 10.0 * torch.log10(se)


# -------- Noise variance (sigma^2) for complex AWGN --------
# Convention:
# - For complex noise n ~ CN(0, sigma^2), we have E[|n|^2] = sigma^2.
# - Each real/imag component has variance sigma^2/2.
# - If average symbol energy Es = 1, then sigma^2 = 1 / SNR_linear.

def snr_db_to_noise_var(snr_db, signal_power=1.0):
    """
    Compute complex noise variance sigma^2 from SNR [dB].
    sigma^2 = Es / SNR_linear. If Es=1, sigma^2 = 10^(-SNR_dB/10).

    Args:
        snr_db (float | Tensor): SNR in dB (per symbol).
        signal_power (float | Tensor): Average symbol energy Es (default 1.0).

    Returns:
        Tensor: sigma^2 (float32), broadcastable to inputs.
    """
    snr_db = torch.as_tensor(snr_db, dtype=torch.float32)
    Es = torch.as_tensor(signal_power, dtype=torch.float32)
    if torch.any(Es <= 0):
        raise ValueError("signal_power must be > 0")
    snr_lin = db2lin(snr_db)
    return Es / snr_lin


def ebn0_db_to_noise_var(ebn0_db, bits_per_symbol, code_rate=1.0, signal_power=1.0):
    """
    Compute complex noise variance sigma^2 from Eb/N0 [dB].

    Relations:
      Es/N0 = Eb/N0 * (bits_per_symbol * code_rate)
      SNR = Es/N0  (when SNR is defined per symbol after equalization)
      sigma^2 = Es / SNR_linear

    Args:
        ebn0_db (float | Tensor): Eb/N0 in dB.
        bits_per_symbol (int | Tensor): Modulation bits per symbol (e.g., QPSK=2).
        code_rate (float | Tensor): Coding rate in (0,1].
        signal_power (float | Tensor): Average symbol energy Es (default 1.0).

    Returns:
        Tensor: sigma^2 (float32).
    """
    ebn0_db = torch.as_tensor(ebn0_db, dtype=torch.float32)
    bps = torch.as_tensor(bits_per_symbol, dtype=torch.float32)
    rate = torch.as_tensor(code_rate, dtype=torch.float32)
    Es = torch.as_tensor(signal_power, dtype=torch.float32)

    if torch.any(bps <= 0):
        raise ValueError("bits_per_symbol must be > 0")
    if torch.any(rate <= 0):
        raise ValueError("code_rate must be > 0")
    if torch.any(Es <= 0):
        raise ValueError("signal_power must be > 0")

    # Es/N0 [dB] = Eb/N0 [dB] + 10*log10(bps*rate)
    esn0_db = ebn0_db + 10.0 * torch.log10(bps * rate)
    return snr_db_to_noise_var(esn0_db, signal_power=Es)


# -------- Convenience: standard deviations for real/imag generation --------

def noise_std_from_sigma2(sigma2):
    """
    For CN(0, sigma^2), each real/imag component ~ N(0, sigma^2/2).
    This returns the per-component std = sqrt(sigma^2/2).
    """
    sigma2 = torch.as_tensor(sigma2, dtype=torch.float32)
    if torch.any(sigma2 < 0):
        raise ValueError("sigma2 must be >= 0")
    return torch.sqrt(sigma2 * 0.5)


# -------- Optional: power utility --------

def average_power(x, dim=None, keepdim=False):
    """
    Estimate average power E[|x|^2] over 'dim' (all dims if None).
    Works for real or complex tensors.
    """
    x = torch.as_tensor(x)
    mag2 = (x.real ** 2 + x.imag ** 2) if torch.is_complex(x) else (x ** 2)
    if dim is None:
        return mag2.mean()
    return mag2.mean(dim=dim, keepdim=keepdim)
