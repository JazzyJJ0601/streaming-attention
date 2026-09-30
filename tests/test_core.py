import numpy as np
from streaming_attn.core import RingAttention


def test_ring_attention_basic():
    """Test basic forward pass."""
    attn = RingAttention(d_model=64, n_heads=4, max_seq=100)
    q = np.random.randn(2, 10, 64)
    k = np.random.randn(2, 10, 64)
    v = np.random.randn(2, 10, 64)
    out = attn.forward(q, k, v)
    assert out.shape == (2, 10, 64)


def test_ring_attention_sliding_window():
    """Test sliding window attention."""
    attn = RingAttention(d_model=64, n_heads=4, max_seq=100)
    attn.set_sliding_window(5)
    q = np.random.randn(2, 10, 64)
    k = np.random.randn(2, 10, 64)
    v = np.random.randn(2, 10, 64)
    out = attn.forward(q, k, v)
    assert out.shape == (2, 10, 64)


def test_ring_attention_cache():
    """Test KV cache ring buffer behavior."""
    attn = RingAttention(d_model=64, n_heads=4, max_seq=5)
    q = np.random.randn(1, 5, 64)
    k = np.random.randn(1, 5, 64)
    v = np.random.randn(1, 5, 64)
    attn.forward(q, k, v, cache=True)
    assert attn.position == 0  # Should wrap around
    
    # Second batch should overwrite
    q2 = np.random.randn(1, 5, 64)
    k2 = np.random.randn(1, 5, 64)
    v2 = np.random.randn(1, 5, 64)
    attn.forward(q2, k2, v2, cache=True)
    assert attn.position == 0


def test_alibi_position_encoding():
    """Test ALiBi bias computation."""
    attn = RingAttention(d_model=64, n_heads=4, max_seq=100)
    bias = attn._alibi_bias(10, 5)  # distance=5
    assert bias.shape == (4,)
    # Further positions should have larger magnitude bias
    bias2 = attn._alibi_bias(10, 0)  # distance=10
    assert np.abs(bias2).sum() > np.abs(bias).sum()


def test_memory_efficiency():
    """Test that KV cache size is fixed regardless of sequence length."""
    max_seq = 10
    attn = RingAttention(d_model=64, n_heads=4, max_seq=max_seq)
    
    # Cache size should be fixed
    assert attn.k_cache.shape == (max_seq, 64)
    assert attn.v_cache.shape == (max_seq, 64)
