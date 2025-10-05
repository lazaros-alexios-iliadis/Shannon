"""
TorchWireless shape and dimension utilities.

Provides:
- Consistent shape checking for PHY tensors (bits, symbols, OFDM grids, channels).
- Human-readable error messages for debugging.
- Batch-safe utilities (for link-level simulations).
"""

import torch


# ----------- Common conventions -----------
# B : batch (simulation instances)
# N_sym : number of OFDM symbols or modulated symbols
# N_sc : number of subcarriers
# N_tx : number of transmit antennas
# N_rx : number of receive antennas
# L : number of paths (for multipath channels)


def check_shape(x: torch.Tensor, expected_shape: tuple | list, name: str = "tensor"):
    """
    Generic shape assertion with wildcard support (-1).

    Example:
        check_shape(x, (B, -1, N_sc))  # any middle dimension
    """
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x)}")

    if len(x.shape) != len(expected_shape):
        raise ValueError(
            f"{name} has {len(x.shape)} dims, expected {len(expected_shape)} ({expected_shape})"
        )

    for i, (got, exp) in enumerate(zip(x.shape, expected_shape)):
        if exp != -1 and got != exp:
            raise ValueError(
                f"{name} dim {i}: expected {exp}, got {got} (shape={tuple(x.shape)})"
            )


def check_batch_dim(x: torch.Tensor, name="tensor"):
    """Ensure batch dimension exists (at least 1)."""
    if x.ndim < 1:
        raise ValueError(f"{name} must include batch dimension (ndim >= 1)")


def check_complex(x: torch.Tensor, name="tensor"):
    """Ensure tensor is complex-valued."""
    if not torch.is_complex(x):
        raise TypeError(f"{name} must be complex-valued (got {x.dtype})")


def check_real(x: torch.Tensor, name="tensor"):
    """Ensure tensor is real-valued."""
    if torch.is_complex(x):
        raise TypeError(f"{name} must be real-valued (got {x.dtype})")


# ----------- PHY-specific checks -----------

def check_bits(x: torch.Tensor, name="bits"):
    """
    Verify bit tensor format.
    Expected dtype: torch.int8 or bool
    Expected values: {0, 1}
    """
    if x.dtype not in (torch.int8, torch.bool):
        raise TypeError(f"{name} must have dtype torch.int8 or torch.bool (got {x.dtype})")
    if not torch.all((x == 0) | (x == 1)):
        raise ValueError(f"{name} must contain only 0 or 1 values (found {x.unique()})")


def check_symbols(x: torch.Tensor, name="symbols"):
    """
    Verify modulated symbol tensor (complex, 2D or 3D).
    Expected shape: [B, N_sym] or [B, N_tx, N_sym]
    """
    if not torch.is_complex(x):
        raise TypeError(f"{name} must be complex-valued.")
    if x.ndim not in (2, 3):
        raise ValueError(f"{name} must have 2 or 3 dims ([B, N_sym] or [B, N_tx, N_sym]), got {x.ndim}")


def check_channel_matrix(H: torch.Tensor, n_rx=None, n_tx=None, name="H"):
    """
    Verify MIMO channel matrix shape: [B, N_rx, N_tx] or [B, N_rx, N_tx, L]
    """
    if not torch.is_complex(H):
        raise TypeError(f"{name} must be complex-valued.")
    if H.ndim not in (3, 4):
        raise ValueError(f"{name} must have 3 or 4 dims ([B, N_rx, N_tx] or [B, N_rx, N_tx, L]).")
    if n_rx is not None and H.shape[1] != n_rx:
        raise ValueError(f"{name} expected {n_rx} RX antennas, got {H.shape[1]}.")
    if n_tx is not None and H.shape[2] != n_tx:
        raise ValueError(f"{name} expected {n_tx} TX antennas, got {H.shape[2]}.")


def check_ofdm_grid(grid: torch.Tensor, name="grid"):
    """
    Verify OFDM grid tensor shape: [B, N_tx or N_rx, N_sym, N_sc]
    """
    if not torch.is_complex(grid):
        raise TypeError(f"{name} must be complex-valued.")
    if grid.ndim != 4:
        raise ValueError(f"{name} must have 4 dims [B, N_ant, N_sym, N_sc], got {grid.shape}")


# ----------- Utility -----------

def shape_str(x: torch.Tensor) -> str:
    """Return a readable string representation of a tensor’s shape and dtype."""
    return f"{tuple(x.shape)} [{x.dtype}] on {x.device}"


def match_batch_dims(*tensors):
    """
    Ensure all tensors share the same batch dimension size.
    """
    batch_sizes = [t.shape[0] for t in tensors if torch.is_tensor(t)]
    if len(set(batch_sizes)) > 1:
        raise ValueError(f"Batch size mismatch across tensors: {batch_sizes}")
