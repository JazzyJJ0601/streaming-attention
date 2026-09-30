# Streaming Attention

A pure NumPy implementation of streaming/ring attention for infinite context windows.

## Features

- **RingAttention class**: Fixed-size KV cache that evicts old tokens using ring buffer logic
- **Sliding window attention**: Restrict attention to recent tokens
- **ALiBi position encoding**: Position-agnostic attention with slope-based bias
- **Pure NumPy**: No external deep learning dependencies

## Usage

```python
from streaming_attn.core import RingAttention

# Create attention module
attn = RingAttention(d_model=512, n_heads=8, max_seq=2048)

# Forward pass
q = np.random.randn(1, 100, 512)
k = np.random.randn(1, 100, 512)
v = np.random.randn(1, 100, 512)
out = attn.forward(q, k, v)
```

## Tests

Run tests with pytest:
```bash
python3 -m pytest tests/ -q
```

## License

MIT
