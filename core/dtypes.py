"""
TorchWireless dtype and complex number utilities.

This module provides standardized dtype handling for all PHY blocks:
  - consistent complex dtype usage (torch.complex64 default)
  - helpers for extracting real dtype
  - device + dtype casting helpers
"""

import torch


# -------- Default global dtypes --------

DEFAULT_COMPLEX_DTYPE = torch.complex64
DEFAULT_FLOAT_DTYPE = torch.float32


# -------- Helper functions --------

def is_complex(x: torch.Tensor) -> bool:
    """Return True if tensor x is complex."""
    return torch.is_complex(x)


def real_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    Return the corresponding real dtype for a given complex dtype.
    Example:
        complex64 -> float32
        complex128 -> float64
        float32 -> float32
    """
    if dtype == torch.complex64:
        return torch.float32
    elif dtype == torch.complex128:
        return torch.float64
    elif dtype.is_floating_point:
        return dtype
    else:
        raise TypeError(f"Unsupported dtype {dtype}")


def complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    Return the corresponding complex dtype for a given real dtype.
    Example:
        float32 -> complex64
        float64 -> complex128
    """
    if dtype == torch.float32:
        return torch.complex64
    elif dtype == torch.float64:
        return torch.complex128
    else:
        raise TypeError(f"Unsupported dtype {dtype}")


def to_device_dtype(x, device=None, dtype=None):
    """
    Move tensor or module to the specified device and dtype if provided.
    Works seamlessly with torch.Tensors or nn.Modules.

    Example:
        x = to_device_dtype(x, device='cuda', dtype=torch.complex64)
    """
    if x is None:
        return None

    if isinstance(x, torch.nn.Module):
        return x.to(device=device, dtype=dtype)
    elif torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    else:
        raise TypeError("Expected torch.Tensor or nn.Module.")


def ensure_complex(x: torch.Tensor, dtype=DEFAULT_COMPLEX_DTYPE) -> torch.Tensor:
    """
    Ensure x is a complex tensor.
    If real, convert to complex by adding zero imaginary part.
    """
    if torch.is_complex(x):
        return x
    return x.to(dtype=real_dtype(dtype)).to(dtype).to(x.device).type(dtype)


def ensure_real(x: torch.Tensor) -> torch.Tensor:
    """Return the real part of a complex tensor, or x if already real."""
    return x.real if torch.is_complex(x) else x


def get_default_dtypes():
    """Return a dict of default dtypes for consistency."""
    return {
        "complex": DEFAULT_COMPLEX_DTYPE,
        "float": DEFAULT_FLOAT_DTYPE,
    }
