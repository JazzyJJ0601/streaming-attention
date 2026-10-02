"""CPU tests for the cache plans in results/run_real.py: budgets hold and the dyadic summaries tile history."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results"))
import run_real as r  # noqa: E402

S = 1100


def entries(raw, pooled):
    n = raw.sum(1)
    if pooled is not None:
        n = n + torch.isfinite(pooled).sum(1)
    return n


def test_dyadic_spans_tile_prefix():
    for n in range(1, 300):
        covered = []
        for k, i in r.dyadic_spans(n):
            covered += list(range(i << k, (i + 1) << k))
        assert covered == list(range(n))
        assert len(r.dyadic_spans(n)) <= n.bit_length()


def test_budgets_hold():
    for C in (64, 128):
        for method in ("window", "sinks"):
            raw, _, pooled = r.plan(S, method, C)
            assert entries(raw, pooled).max() == C
        for B in (4, 16):
            raw, levels, pooled = r.plan(S, "dyadic", C, B)
            assert entries(raw, pooled).max() <= C


def test_dyadic_covers_every_past_token_once():
    B, C = 8, 96
    raw, levels, pooled = r.plan(S, "dyadic", C, B)
    for q in (0, 5, 100, 500, S - 1):
        count = raw[q, :q + 1].long().clone()
        for col in torch.nonzero(torch.isfinite(pooled[q]))[:, 0].tolist():
            k, i = levels[col]
            count[r.SINKS + (i << k) * B:r.SINKS + ((i + 1) << k) * B] += 1
        assert torch.equal(count, torch.ones(q + 1, dtype=torch.long)), q
        assert not raw[q, q + 1:].any()


def test_pool_keys_matches_span_means():
    B = 4
    x = torch.randn(1, 2, 4 + 8 * B, 3)
    levels = [(0, 1), (1, 1), (3, 0)]
    p = r.pool_keys(x, levels, B)
    assert torch.allclose(p[:, :, 0], x[:, :, 4 + B:4 + 2 * B].mean(2))
    assert torch.allclose(p[:, :, 1], x[:, :, 4 + 2 * B:4 + 4 * B].mean(2))
    assert torch.allclose(p[:, :, 2], x[:, :, 4:4 + 8 * B].mean(2), atol=1e-6)


def _attend(q, k, v, alpha, beta):
    r.STATE.update(method="dyadic", B=4, alpha=alpha, beta=beta, plan=r.plan(q.shape[2], "dyadic", 24, 4))
    return r.stream_attention(None, q, k, v, None)[0]


def test_variance_term_vanishes_for_constant_spans():
    torch.manual_seed(0)
    S = 4 + 4 * 16
    q, v = torch.randn(1, 4, S, 8), torch.randn(1, 2, S, 8)
    k = torch.randn(1, 2, S, 8)
    k[:, :, 4:] = k[:, :, 4:5]                                    # every pooled span has zero variance
    r.STATE["plan"] = None
    a = _attend(q, k, v, 1.0, 0.0)
    b = _attend(q, k, v, 1.0, 1.0)
    assert torch.allclose(a, b, atol=1e-5)


def test_alpha_zero_and_variance_change_output():
    torch.manual_seed(1)
    S = 4 + 4 * 16
    q, k, v = torch.randn(1, 4, S, 8), torch.randn(1, 2, S, 8), torch.randn(1, 2, S, 8)
    base = _attend(q, k, v, 1.0, 0.0)
    assert not torch.allclose(base, _attend(q, k, v, 0.0, 0.0), atol=1e-4)
    assert not torch.allclose(base, _attend(q, k, v, 1.0, 1.0), atol=1e-4)


def test_exact_budget_is_full_and_tiles_history():
    B, C = 8, 96
    raw, levels, pooled = r.plan(S, "dyadic", C, B, exact=True)
    n = entries(raw, pooled)
    assert n.max() == C
    assert n[C:].min() >= C - B - ((S - r.SINKS) // B).bit_length()  # no worst-case reserve kept back
    for q in (0, 100, 500, S - 1):
        count = raw[q, :q + 1].long().clone()
        for col in torch.nonzero(torch.isfinite(pooled[q]))[:, 0].tolist():
            k, i = levels[col]
            count[r.SINKS + (i << k) * B:r.SINKS + ((i + 1) << k) * B] += 1
        assert torch.equal(count, torch.ones(q + 1, dtype=torch.long)), q


def test_rope_round_trip_and_recentring():
    torch.manual_seed(2)
    r.STATE["inv_freq"] = 1.0 / (1e6 ** (torch.arange(0, 8, 2).float() / 8))
    x = torch.randn(1, 2, 20, 8)
    pos = torch.arange(20)
    assert torch.allclose(r._rot(r._rot(x, pos, 1), pos, -1), x, atol=1e-5)
    # a span whose pre-RoPE keys are identical summarises to that key rotated at the span centre
    k0 = torch.randn(8)
    key = r._rot(k0.expand(1, 1, 20, 8).clone(), pos, 1)
    r.STATE["rope"] = True
    s = r.summary_keys(key, [(0, 0)], 4)                           # tokens 4..7, centre 5.5
    r.STATE["rope"] = False
    assert torch.allclose(s[0, 0, 0], r._rot(k0[None], torch.tensor([5.5]), 1)[0], atol=1e-5)
