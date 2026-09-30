import numpy as np


class RingAttention:
    """
    RingAttention implements streaming/ring attention for infinite context length.
    
    Features:
    - Fixed-size KV cache using ring buffer (evicts old tokens by overwriting)
    - Sliding window attention support
    - ALiBi position encoding
    """
    
    def __init__(self, d_model: int, n_heads: int, max_seq: int,
                 alibi_slope_min: float = -0.0001,
                 alibi_slope_max: float = -1.0):
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq = max_seq
        self.d_head = d_model // n_heads
        
        # Initialize ALiBi slopes (negative slopes as per ALiBi paper)
        slopes = np.exp(np.linspace(alibi_slope_min, alibi_slope_max, n_heads))
        self.alibi_slopes = slopes
        
        # KV cache: ring buffer storing [max_seq, d_model] (flattened per token)
        self.k_cache = np.zeros((max_seq, d_model))
        self.v_cache = np.zeros((max_seq, d_model))
        
        # Track current position in ring buffer
        self.position = 0
        
        # Sliding window size (None means full context)
        self.window_size = None
    
    def set_sliding_window(self, size: int):
        """Set sliding window size for restricted attention."""
        self.window_size = size
    
    def _alibi_bias(self, query_pos: int, key_pos: int) -> np.ndarray:
        """Compute ALiBi position encoding bias."""
        distance = query_pos - key_pos
        bias = self.alibi_slopes * distance
        return bias  # [n_heads]
    
    def forward(self, q: np.ndarray, k: np.ndarray, v: np.ndarray,
                cache: bool = True) -> np.ndarray:
        """
        Forward pass for attention.
        
        Args:
            q: Query tensor [batch, seq_len, d_model]
            k: Key tensor [batch, seq_len, d_model]
            v: Value tensor [batch, seq_len, d_model]
            cache: Whether to store KV in ring buffer
            
        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = q.shape
        
        # Get current positions in cache
        start_pos = self.position
        
        # Store in KV cache (ring buffer)
        if cache:
            for i in range(seq_len):
                cache_idx = (start_pos + i) % self.max_seq
                self.k_cache[cache_idx] = k[0, i, :]  # Use batch 0 for cache
                self.v_cache[cache_idx] = v[0, i, :]
        
        # Initialize output
        output = np.zeros_like(q)
        
        # For each batch
        for b in range(batch):
            for i in range(seq_len):
                query_pos = (start_pos + i) % self.max_seq
                query_vec = q[b, i, :]
                
                # Compute attention scores against all cached positions
                scores = np.zeros(self.max_seq)
                
                for j in range(self.max_seq):
                    # Dot product between query and cached key
                    scores[j] = np.dot(query_vec, self.k_cache[j]) / np.sqrt(self.d_head)
                    
                    # Add ALiBi bias (average over heads)
                    key_pos = j
                    bias = self._alibi_bias(query_pos, key_pos).mean()
                    scores[j] += bias
                    
                    # Apply sliding window mask
                    if self.window_size is not None:
                        distance = (query_pos - key_pos) % self.max_seq
                        if distance > self.window_size:
                            scores[j] = -1e9
                
                # Softmax
                scores = scores - scores.max()
                scores = np.exp(scores)
                scores = scores / scores.sum()
                
                # Compute output as weighted sum of cached values
                out_vec = np.zeros(d_model)
                for j in range(self.max_seq):
                    out_vec += scores[j] * self.v_cache[j]
                
                output[b, i, :] = out_vec
        
        # Advance position
        self.position = (self.position + seq_len) % self.max_seq
        
        return output


if __name__ == "__main__":
    # Simple test
    attn = RingAttention(d_model=64, n_heads=4, max_seq=100)
    q = np.random.randn(2, 10, 64)
    k = np.random.randn(2, 10, 64)
    v = np.random.randn(2, 10, 64)
    out = attn.forward(q, k, v)
    print(f"Output shape: {out.shape}")
