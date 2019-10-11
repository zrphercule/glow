from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch_glow

from tests.utils import jitVsGlow


def test_quantized_add_zerooffset():
    """Basic test of the PyTorch quantized::add Node on Glow with zero offset."""

    def test_f(a, b):
        q = torch.nn.quantized.Quantize(
            scale=0.3, zero_point=0, dtype=torch.quint8)
        return torch.ops.quantized.add(q(a), q(b), scale=0.05, zero_point=0)

    x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    y = torch.tensor([5, 6, 7, 8], dtype=torch.float32)

    #jitVsGlow(test_f, x, y, expected_fused_ops={"quantized::add",
    #                                            "aten::quantize_per_tensor",
    #                                            "aten::dequantize"})
    with torch.no_grad():
        torch_glow.disableFusionPass()
        traced = torch.jit.trace(test_f, (x, y))
        res = traced(x, y)
    print(traced.graph)
