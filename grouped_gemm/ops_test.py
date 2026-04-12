import unittest
import itertools

from absl.testing import parameterized
from grouped_gemm import ops
import grouped_gemm
import numpy as np
import torch


def allclose(x, y, pct=2.0):
    mask = torch.isclose(x, y, rtol=1e-5)
    pct_diff = (mask.numel() - mask.sum()) / mask.numel() * 100
    if pct_diff > pct:
        print(x[torch.logical_not(mask)], y[torch.logical_not(mask)])
        print("{:.2f}% of values not close.".format(pct_diff))
        return False
    return True


def add_transpose_flags(x):
    out = []
    for y in x:
        for f in [(False,), (True,)]:
            out.append(y + f)
    return out


_TEST_PROBLEMS = add_transpose_flags((
    (1, 128, 128, 128),
    (8, 128, 128, 128),
    (16, 128, 128, 128),
    (1, 128, 256, 512),
    (8, 128, 256, 512),
    (16, 128, 256, 512),
))


def randn(bs, x, y, dtype=torch.bfloat16):
    if bs >= 1:
        out = (torch.rand(bs, x, y) - 0.5 * 2) / (y * x)
    else:
        out = (torch.rand(x, y) - 0.5 * 2) / (y * x)
    return out.cuda().to(dtype)


def gmm(a, b, batch_sizes, trans_b=False):
    batch_sizes = batch_sizes.numpy()

    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        if isinstance(b, list):
            rhs = b[i][:, :].t() if trans_b else b[i][:, :]
        else:
            rhs = b[i, :, :].t() if trans_b else b[i, :, :]
        out.append(a[start:start + size, :] @ rhs)
        start += size
    return torch.cat(out)

def gmmv(a, b, c, batch_sizes, trans_b=False):
    batch_sizes = batch_sizes.numpy()

    out = []
    start = 0
    for i, size in enumerate(batch_sizes):
        rhs = b[start:start + size, :].t() if trans_b else b[start:start + size, :]
        if isinstance(c, list):
            out.append((a[start:start + size, :].t() @ rhs) + c[i][:, :])
            return out
        else:
            out.append((a[start:start + size, :].t() @ rhs) + c[i, :, :])
        start += size
    return torch.cat(out)


@parameterized.parameters(*_TEST_PROBLEMS)
class OpsTest(parameterized.TestCase):

    def testGroupedGemm_FixedSizes(self, z, m, k, n, trans_b):
        torch.manual_seed(0)
        a = randn(z, m, k).view(-1, k)
        b = randn(z, n, k) if trans_b else randn(z, k, n)
        batch_sizes = torch.tensor([m] * z)

        a.requires_grad_(True)
        b.requires_grad_(True)
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        out = ops.gmm(a, b, batch_sizes, trans_b)
        expected_out = gmm(a_ref, b_ref, batch_sizes, trans_b)
        self.assertTrue(allclose(out, expected_out))

        # Check gradients.
        out.sum().backward()
        expected_out.sum().backward()
        self.assertTrue(allclose(a.grad, a_ref.grad))
        self.assertTrue(allclose(b.grad, b_ref.grad))

    def testGroupedGemm_FixedSizesList(self, z, m, k, n, trans_b):
        torch.manual_seed(0)
        a = randn(z, m, k).view(-1, k)
        b = [randn(0, n, k) if trans_b else randn(0, k, n) for _ in range(z)]
        batch_sizes = torch.tensor([m] * z)

        a.requires_grad_(True)
        for i in range(z):
            b[i].requires_grad_(True)
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = [i.detach().clone().requires_grad_(True) for i in b]

        out = grouped_gemm.grouped_gemm.backend.gmmfwd(
            a, b, batch_sizes, 
            trans_a=False,
            trans_b=trans_b
        )
        expected_out = gmm(a_ref, b_ref, batch_sizes, trans_b)
        self.assertTrue(allclose(out, expected_out))

    def testGroupedGemm_VariableSizes(self, z, m, k, n, trans_b):
        torch.manual_seed(0)
        a = randn(z, m, k).view(-1, k)
        b = randn(z, n, k) if trans_b else randn(z, k, n)

        dist = torch.rand(z, )
        dist /= dist.sum()
        batch_sizes = (dist * m).to(torch.long)
        error = m * z - batch_sizes.sum()
        batch_sizes[-1] += error
        assert batch_sizes.sum() == (m * z)

        a.requires_grad_(True)
        b.requires_grad_(True)
        a_ref = a.detach().clone().requires_grad_(True)
        b_ref = b.detach().clone().requires_grad_(True)

        out = ops.gmm(a, b, batch_sizes, trans_b)
        expected_out = gmm(a_ref, b_ref, batch_sizes, trans_b)
        self.assertTrue(allclose(out, expected_out))

        # Check gradients.
        out.sum().backward()
        expected_out.sum().backward()
        self.assertTrue(allclose(a.grad, a_ref.grad))
        self.assertTrue(allclose(b.grad, b_ref.grad))

    def testGroupedGemm_VariableSizesFP32(self, z, m, k, n, trans_b):
        torch.manual_seed(0)
        a = randn(z, m, k).view(-1, k)
        b = randn(z, m, n).view(-1, n)
        c = randn(z, k, n, dtype=torch.float32)

        dist = torch.rand(z, )
        dist /= dist.sum()
        batch_sizes = (dist * m).to(torch.long)
        error = m * z - batch_sizes.sum()
        batch_sizes[-1] += error
        assert batch_sizes.sum() == (m * z)

        a_ref = a.detach().clone()
        b_ref = b.detach().clone()
        c_ref = c.detach().clone()

        out = grouped_gemm.grouped_gemm.backend.gmm(
            a, 
            b,
            batch_sizes,
            trans_a=True,
            trans_b=False,
            c = c,
            alpha=1.0,
            beta=1.0
        )
        
        expected_out = gmmv(a_ref, b_ref, c_ref, batch_sizes, False)
        self.assertTrue(allclose(out, expected_out.view(out.shape)))

    def testGroupedGemm_VariableSizesFP32List(self, z, m, k, n, trans_b):
        torch.manual_seed(0)
        a = randn(z, m, k).view(-1, k)
        b = randn(z, m, n).view(-1, n)
        c = [randn(0, k, n, dtype=torch.float32) for _ in range(z)]

        dist = torch.rand(z, )
        dist /= dist.sum()
        batch_sizes = (dist * m).to(torch.long)
        error = m * z - batch_sizes.sum()
        batch_sizes[-1] += error
        assert batch_sizes.sum() == (m * z)

        a_ref = a.detach().clone()
        b_ref = b.detach().clone()
        c_ref = [i.detach().clone() for i in c]

        out = grouped_gemm.grouped_gemm.backend.gmmbwd(
            a, 
            b,
            batch_sizes,
            trans_a=True,
            trans_b=False,
            c = c,
            alpha=1.0,
            beta=1.0
        )
        
        expected_out = gmmv(a_ref, b_ref, c_ref, batch_sizes, False)
        for o, e in zip(out, expected_out):
            self.assertTrue(allclose(o, e.view(o.shape)))


if __name__ == '__main__':
    unittest.main()
