# Streaming Attention: do summaries of evicted tokens help? (A negative result.)

A bounded KV cache for long text on Qwen3-8B. Each head keeps at most C entries. I tested whether
**mean-pooled summaries of evicted tokens** (sinks + recent window + a binary-counter hierarchy of block
summaries) beat StreamingLLM's attention sinks + recent window. After three versions, **they do not**. The best
version comes within 0.2–1% of StreamingLLM but never beats it, on either budget, on train or test.

Perplexity on WikiText-2 test (16 documents × 4,096 tokens, scored on tokens 1,024–4,095 where the cache is
already full; lower is better). Full attention: **8.72**.

| Method | C = 256 | C = 512 |
|---|---:|---:|
| Recent window only | 26.58 | 19.40 |
| **StreamingLLM (4 sinks + window), the baseline** | **11.36** | **10.07** |
| Ours v1: dyadic summaries, +log(m) bias, B=16 | 12.42 | 10.55 |
| Ours v1, B=64 | 12.75 | 10.64 |
| Ours v1, no bias | 11.52 | 10.12 |
| Ours v2: bias scale and variance term chosen on train | 11.51 | 10.12 |
| Ours v3: exact budget (+ RoPE re-centring tried), chosen on train | 11.49 | 10.10 |

All settings for v2 and v3 were chosen on WikiText-2 **train** text, then scored once on test. On train, sinks
also won every time (13.31 vs 13.40 at C=256, 11.78 vs 11.80 at C=512).

## The method

Tokens that fall out of the window are pooled into blocks of B tokens. Blocks merge like a binary counter
(aligned spans of 1, 2, 4, … blocks), so at most log2(blocks)+1 summary slots exist at once. Each summary is
the mean key and mean value of its span. It is attended with a score bias of α·log(m), so one slot can carry
the softmax mass of the m tokens it replaces.

## What I tried, and why it didn't work

1. **v1:** the full +log(m) bias hurt badly (12.42 vs 11.36). Big summaries soaked up attention that belonged
   to specific tokens. Dropping the bias brought it close (11.52).
2. **v2:** I tuned the bias scale α and added a second-order term for the spread of keys inside a span (β),
   both chosen on train. Small gain (11.51).
3. **v3:** I fixed the two flaws I could find:
   - **Reserved slots.** v1/v2 shrank the window so the worst case fit in C, leaving about 12 of 256 slots
     unused on average. v3 pools blocks only while the cache is over budget, so the cache stays full.
   - **Rotation blurring.** Averaging keys after RoPE cancels the fast-rotating dimensions. v3 tried
     un-rotating keys, averaging, and re-rotating at the span centre.

   The exact budget gave a tiny gain (11.49). RoPE re-centring changed nothing on train (13.434 vs 13.424 at
   C=256), so it wasn't chosen.

**Why it loses:** Qwen3-8B on this text gets almost all the value of the past from the sinks and the most
recent tokens. A mean of hundreds of old keys is too blurred to match any one query well. So every slot it
takes from the window costs more than it returns. A summary would need to be *selective* (keep the specific
tokens that get attended), which is a different method. That is what
[predictive-swap](https://github.com/JazzyJJ0601/predictive-swap) does, and it beats sinks (8.89 vs 9.99 at
C=512).

## Run it

```bash
python results/run_real.py v1    # or v2 / v3; needs Qwen3-8B locally (path at the top of the script)
python -m pytest -q tests        # span bookkeeping, exact budget, RoPE round trip
```

Raw numbers: `results/real.json`, `real_v2.json`, `real_v3.json`. Details: [RESULTS.md](RESULTS.md).

*An earlier version of this repo was a NumPy ring-buffer toy with a GPT-2 baseline. It was replaced by this
real-model study.*
