from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from torch.nn.modules.utils import _pair

from tests.utils import jitVsGlow


def test_quantized_conv2d():
    """Basic test of the PyTorch quantized::add Node on Glow."""

    def test_f(a, w, b):

        q = torch.nn.quantized.Quantize(1/128, 5, torch.quint8)
        q2 = torch.nn.quantized.Quantize(1/128, 5, torch.qint8)
        qa = q(a)
        qw = q2(w)
        #qb = q2(b)

        stride = _pair(1)
        padding = _pair(0)
        dilation = _pair(1)
        scale = 1/128
        offset = 0
        groups = 1


        prepacked_weight = torch.ops.quantized.conv_prepack(qw, b, stride,
                                                            padding, dilation,
                                                            groups)

        conv = torch.ops.quantized.conv2d(qa, prepacked_weight, stride,
                                          padding, dilation, groups, scale,
                                          offset)
        dq = torch.nn.quantized.DeQuantize()
        return dq(conv)

    x = torch.randn([1, 5, 5, 3])
    w = torch.randn([1, 3, 3, 3])
    b = torch.randn(1)

    jitVsGlow(test_f, x, w, b, expected_fused_ops={"quantized::add",
                                                "aten::quantize_linear",
                                                "aten::dequantize"})
