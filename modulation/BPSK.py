from torch import Tensor
import torch


def bpsk(u_coded_bits: Tensor, complex_out: bool = True) -> Tensor:
    if u_coded_bits.dtype not in (torch.int8, torch.bool):
        raise TypeError(f"bpsk() expects bits as int8 or bool, got {u_coded_bits.dtype}")

        # Ensure bits are in {0,1}
    if not torch.all((u_coded_bits==0) | (u_coded_bits==1)):
        raise ValueError("bpsk() expects binary bits {0,1}")

    s_mod = 1.0 - 2.0 * u_coded_bits.to(torch.float32)  # 0->+1, 1->-1

    if complex_out:
        s_mod = s_mod.to(torch.complex64)
    return s_mod
