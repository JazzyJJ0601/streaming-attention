# Results: dyadic summary cache vs StreamingLLM on Qwen3-8B

- **Model:** Qwen3-8B, original RoPE positions; documents fit inside its trained context.
- **Test:** WikiText-2 test, 16 × 4,096 tokens, scored on tokens 1,024–4,095.
- **Selection:** WikiText-2 train, 4 × 4,096 tokens. 4 sinks, block B = 16 unless stated.
- **Full attention:** 8.7155.

## Test perplexity

| Method | C = 256 | C = 512 |
|---|---:|---:|
| window | 26.5756 | 19.4038 |
| **sinks (StreamingLLM)** | **11.3559** | **10.0735** |
| v1 dyadic, B=16, +log m bias | 12.4206 | 10.5536 |
| v1 dyadic, B=64, +log m bias | 12.7469 | 10.6397 |
| v1 dyadic, no bias | 11.5239 | 10.1194 |
| v2 (chosen: α=0.25, β=1 at 256; α=0, β=0 at 512) | 11.5104 | 10.1194 |
| v3 exact budget (chosen: rope off; α=0.25, β=1 at 256; α=0, β=0 at 512) | 11.4912 | 10.0956 |

Gap of the best version to sinks: +0.135 (+1.2%) at C=256, +0.022 (+0.2%) at C=512.

## Train selection grids (perplexity)

**v2** (worst-case reserve): sinks 13.3084 / 11.7763.

| α, β | C=256 | C=512 |
|---|---:|---:|
| 0, 0 | 13.4340 | 11.8356 |
| 0, 1 | 13.4246 | 11.8482 |
| 0.25, 0 | 13.4289 | 11.8416 |
| 0.25, 1 | 13.4008 | 11.8397 |
| 0.5, 0 | 13.4989 | 11.8559 |
| 0.5, 1 | 13.5068 | 11.8737 |
| 1, 0 | 14.4222 | 12.2878 |
| 1, 1 | 15.6215 | 13.1048 |

**v3** (exact budget): sinks 13.3084 / 11.7763.

| RoPE re-centred, α, β | C=256 | C=512 |
|---|---:|---:|
| off, 0, 0 | 13.4243 | 11.8046 |
| off, 0, 1 | 13.4262 | 11.8149 |
| off, 0.25, 0 | 13.4356 | 11.8091 |
| off, 0.25, 1 | 13.3998 | 11.8157 |
| off, 0.5, 0 | 13.4929 | 11.8257 |
| off, 0.5, 1 | 13.5006 | 11.8691 |
| on, 0 | 13.4342 | 11.8053 |
| on, 0.25 | 13.4388 | 11.8132 |
| on, 0.5 | 13.4913 | 11.8248 |

## Conclusion

No configuration beats attention sinks, on train or test, at either budget. The exact budget helps a little.
RoPE re-centring and the variance term make no consistent difference. Mean-pooled summaries of evicted tokens
are worth less than the recent tokens they displace. Not cherry-picked: every configuration run is listed
above.
