"""Real Qwen3-8B run: bounded-cache streaming attention vs full attention on long text.

Every method keeps at most C cache entries per head while reading a 4096-token document; perplexity is
scored on tokens 1024..4095 (where the cache is already full), WikiText-2 test, 16 documents.

  full       ordinary causal attention over everything (the upper bound)
  window     the last C tokens only
  sinks      StreamingLLM: the first 4 tokens ("attention sinks") + the last C-4 tokens
  dyadic     ours: sinks + a recent window + mean-pooled summaries of every evicted token. Evicted tokens are
             pooled in base blocks of B tokens; blocks merge like a binary counter (aligned spans of 1, 2, 4,
             ... blocks), so at most log2(n_blocks)+1 summary slots exist at any time. A summary of m tokens
             is attended with a +log(m) score bias, so one slot carries roughly the softmax mass of the m
             tokens it replaces. The window is shrunk so that the WORST-CASE entry count equals C.

All methods use the model's original RoPE positions (documents are inside Qwen3's trained context, so
StreamingLLM's position re-indexing is not needed here). Keys are pooled after RoPE, as a cache stores them.
Modes (argv[1]): v1 (above; writes real.json), v2 (bias scale alpha and a key-variance term beta chosen on WikiText-2
train; real_v2.json), v3 (exact budget: blocks are pooled only while the cache is over C, so no slots are held in
reserve; optionally RoPE re-centred summaries: keys are un-rotated, averaged, and re-rotated at the span centre;
chosen on train; real_v3.json).
"""
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_PATH = "/home/jasper/eirene-projects/03-inference-lab/ai-lab/models/Qwen--Qwen3-8B"
SEQ, SCORE_FROM, N_DOCS, SINKS = 4096, 1024, 16, 4
BUDGETS = (256, 512)
OUT = Path(__file__).resolve().parent / "real.json"
STATE = {"method": "full", "alpha": 1.0, "beta": 0.0, "rope": False, "inv_freq": None}


def dyadic_spans(n):
    """Aligned dyadic decomposition of the first n base blocks, oldest (largest) first: [(level, index)]."""
    spans, off = [], 0
    for k in reversed(range(max(n, 1).bit_length())):
        if n >> k & 1:
            spans.append((k, off >> k))
            off += 1 << k
    return spans


def plan(S, method, C, B=16, bias=True, exact=False):
    """Which raw keys and which pooled spans each query sees. Returns (allowed raw (S,S) bool,
    span list [(level, index)], pooled mask (S, n_spans) float with -inf where not visible).
    exact=False reserves the worst case (window shrunk by max slots + B-1); exact=True pools a block only when
    the cache would otherwise exceed C, so every query uses exactly min(q+1, C) entries, as a real cache would."""
    t = torch.arange(S)[:, None]
    j = torch.arange(S)[None, :]
    causal = j <= t
    if method == "full":
        return causal, [], None
    if method == "window":
        return causal & (j > t - C), [], None
    if method == "sinks":
        return causal & ((j < SINKS) | (j > t - (C - SINKS))), [], None
    # dyadic: worst case entries = SINKS + max_slots + (W + B - 1)
    nb = (S - SINKS) // B
    W = C - SINKS - nb.bit_length() - (B - 1)
    assert W > 0, (C, B)
    levels = [(k, i) for k in range(nb.bit_length()) for i in range(nb >> k)]
    idx = {s: n for n, s in enumerate(levels)}
    raw = torch.zeros(S, S, dtype=torch.bool)
    pooled = torch.full((S, len(levels)), float("-inf"))
    n = 0
    for q in range(S):
        if exact:                               # evict (pool) whole blocks only while over budget
            while n < nb and min(SINKS, q + 1) + bin(n).count("1") + max(0, q + 1 - SINKS - n * B) > C:
                n += 1
        else:
            frontier = max(SINKS, q - W + 1)    # tokens before this have left the plain window
            n = min((frontier - SINKS) // B, nb)  # whole base blocks evicted so far
        lo = SINKS + n * B                      # raw window starts at the first un-pooled token
        raw[q, :min(SINKS, q + 1)] = True
        raw[q, lo:q + 1] = True
        for k, i in dyadic_spans(n):
            pooled[q, idx[(k, i)]] = math.log(B << k) if bias else 0.0
    return raw, levels, pooled


def pool_keys(x, levels, B):
    """x: (b, h, S, D) -> (b, h, len(levels), D) means of the aligned spans of base blocks."""
    b, h, S, D = x.shape
    nb = (S - SINKS) // B
    base = x[:, :, SINKS:SINKS + nb * B].float().reshape(b, h, nb, B, D).mean(3)
    out = []
    for k, i in levels:
        out.append(base[:, :, i << k:(i + 1) << k].mean(2))
    return torch.stack(out, 2).to(x.dtype)


def _rot(x, pos, sign):
    """Rotate x (..., S, D) by RoPE angles at (possibly fractional) positions pos (S,); sign=-1 undoes it."""
    f = pos.to(x.device, torch.float32)[:, None] * STATE["inv_freq"].to(x.device, torch.float32)[None]
    cos, sin = torch.cat([f, f], -1).cos(), sign * torch.cat([f, f], -1).sin()
    h = x.shape[-1] // 2
    return x * cos + torch.cat([-x[..., h:], x[..., :h]], -1) * sin


def summary_keys(key, levels, B):
    """Mean-pooled summary keys. rope=True: undo RoPE, average, and re-apply it at each span's centre, so
    high-frequency rotary dims don't cancel out in the mean."""
    if not STATE["rope"]:
        return pool_keys(key, levels, B)
    S = key.shape[2]
    pre = _rot(key.float(), torch.arange(S), -1)
    centre = torch.tensor([SINKS + ((i << k) * B) + ((B << k) - 1) / 2 for k, i in levels])
    return _rot(pool_keys(pre, levels, B), centre, 1).to(key.dtype)


def stream_attention(module, query, key, value, attention_mask, scaling=None, **kw):
    S = query.shape[2]
    m = STATE["method"]
    if m == "full":
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scaling, enable_gqa=True)
        return out.transpose(1, 2).contiguous(), None
    raw, levels, pooled = STATE["plan"]
    mask = torch.where(raw.to(query.device), 0.0, float("-inf"))
    mask = mask.to(query.dtype)[None, None]
    if levels:
        B, alpha, beta = STATE["B"], STATE["alpha"], STATE["beta"]
        pm = pooled.to(query.device)
        pm = torch.where(torch.isfinite(pm), pm * alpha, pm)[None, None]       # bias = alpha * log(m)
        if beta and not STATE["rope"]:
            # E[exp(s q.k)] ~ exp(s q.mu + s^2/2 q^T diag(var) q): add the second-order term the mean drops
            s2 = (scaling if scaling is not None else query.shape[-1] ** -0.5) ** 2
            kf = key.float()
            var = (pool_keys(kf * kf, levels, B) - pool_keys(kf, levels, B) ** 2).clamp_min(0)
            var = var.repeat_interleave(query.shape[1] // key.shape[1], 1)
            pm = pm + beta * 0.5 * s2 * (query.float() ** 2) @ var.transpose(-1, -2)
        key = torch.cat([key, summary_keys(key, levels, B)], 2)
        value = torch.cat([value, pool_keys(value, levels, B)], 2)
        H = query.shape[1]
        mask = torch.cat([mask.expand(1, H, S, S), pm.to(query.dtype).expand(1, H, S, pm.shape[-1])], 3)
    out = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scaling, enable_gqa=True)
    return out.transpose(1, 2).contiguous(), None


@torch.no_grad()
def nll(model, docs):
    """Summed NLL and token count over positions SCORE_FROM..SEQ-1 (lm_head in chunks to save memory)."""
    tot, cnt = 0.0, 0
    for d in docs:
        x = d.unsqueeze(0).cuda()
        h = model.model(input_ids=x).last_hidden_state[0]
        for s in range(SCORE_FROM - 1, SEQ - 1, 512):
            e = min(s + 512, SEQ - 1)
            logp = torch.log_softmax(model.lm_head(h[s:e]).float(), -1)
            tot -= logp.gather(1, x[0, s + 1:e + 1, None]).sum().item()
            cnt += e - s
    return tot, cnt


def main():
    from datasets import load_dataset
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
    AttentionInterface.register("stream", stream_attention)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, local_files_only=True,
                                                 device_map="cuda").eval()
    model.set_attn_implementation("stream")
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    docs = [ids[i * SEQ:(i + 1) * SEQ] for i in range(N_DOCS)]
    res = {"model": "Qwen3-8B", "data": f"wikitext-2 test, {N_DOCS} x {SEQ} tokens",
           "scored": f"tokens {SCORE_FROM}..{SEQ - 1}", "sinks": SINKS, "rows": []}

    STATE["inv_freq"] = model.model.rotary_emb.inv_freq.detach().float().cpu()
    mode = sys.argv[1] if len(sys.argv) > 1 else "v1"
    v2 = mode in ("v2", "v3")
    out = OUT.with_name(f"real_{mode}.json") if v2 else OUT

    def run(method, C=None, B=16, bias=True, label=None, alpha=1.0, beta=0.0, on=None, record=True,
            exact=False, rope=False):
        STATE["method"], STATE["B"], STATE["alpha"], STATE["beta"], STATE["rope"] = method, B, alpha, beta, rope
        STATE["plan"] = plan(SEQ, method, C, B, bias, exact) if method != "full" else None
        t = time.time()
        tot, cnt = nll(model, docs if on is None else on)
        row = {"method": label or method, "budget": C, "ppl": round(math.exp(tot / cnt), 4), "tokens": cnt,
               "seconds": round(time.time() - t, 1)}
        if method == "dyadic":
            nb = (SEQ - SINKS) // B
            row.update(block=B, bias=bias, alpha=alpha, beta=beta,
                       window=C - SINKS - nb.bit_length() - (B - 1), max_summary_slots=nb.bit_length())
            if exact:
                row.update(exact_budget=True, rope_recentred=rope, window="C - 4 - live summary slots")
        if on is not None:
            row["data"] = "train (selection)"
        print(row, flush=True)
        if record:
            res["rows"].append(row)
            out.write_text(json.dumps(res, indent=2))
        return row["ppl"]

    if mode == "v3":
        # v3: exact-budget eviction (no worst-case reserve) +/- RoPE re-centred summaries; chosen on TRAIN
        tr = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"])
        tids = tok(tr[:2_000_000], return_tensors="pt").input_ids[0]
        train = [tids[i * SEQ:(i + 1) * SEQ] for i in range(4)]
        res["selection"] = ("exact budget; rope in {off,on}, alpha in {0,0.25,0.5}, beta in {0,1} (rope off "
                            "only); chosen on 4 x 4096 train tokens per budget")
        cfgs = [(ro, a, b) for ro in (False, True) for a in (0.0, 0.25, 0.5) for b in ((0.0, 1.0) if not ro else (0.0,))]
        for C in BUDGETS:
            grid = {c: run("dyadic", C, alpha=c[1], beta=c[2], rope=c[0], exact=True, on=train) for c in cfgs}
            ro, a, b = min(grid, key=grid.get)
            run("sinks", C, on=train)
            run("sinks", C)
            run("dyadic", C, alpha=a, beta=b, rope=ro, exact=True, label="dyadic-v3")
        return
    if v2:
        # v2: choose the summary bias scale alpha and variance weight beta on WikiText-2 TRAIN, then score on test
        tr = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"])
        tids = tok(tr[:2_000_000], return_tensors="pt").input_ids[0]
        train = [tids[i * SEQ:(i + 1) * SEQ] for i in range(4)]
        res["selection"] = "alpha in {0,0.25,0.5,1}, beta in {0,1}, chosen on 4 x 4096 train tokens per budget"
        for C in BUDGETS:
            grid = {(a, b): run("dyadic", C, alpha=a, beta=b, on=train, record=True)
                    for a in (0.0, 0.25, 0.5, 1.0) for b in (0.0, 1.0)}
            a, b = min(grid, key=grid.get)
            run("sinks", C, on=train, record=True)
            run("dyadic", C, alpha=a, beta=b, label="dyadic-v2")
        return

    run("full")
    for C in BUDGETS:
        run("window", C)
        run("sinks", C)
        run("dyadic", C, B=16)
        run("dyadic", C, B=64)
        run("dyadic", C, B=16, bias=False, label="dyadic-nobias")


if __name__ == "__main__":
    main()
