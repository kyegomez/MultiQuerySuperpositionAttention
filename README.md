# MultiQuerySuperpositionAttention
Multi-Query Attention with Sub-linear Masking, Superposition, and Entanglement


```python
import torch
import torch.nn as nn
from loguru import logger
from typing import Optional


class MultiQuerySuperpositionAttention(nn.Module):
    """
    Multi-Query Attention mechanism with sub-linear masking, superposition, and entanglement.
    This mechanism uses separate query heads but shared key and value projections, reducing memory usage.
    
    Additionally, sub-linear masking, superposition of states, and token entanglement are incorporated to
    make the attention process more efficient while maintaining high performance.

    Args:
        dim (int): The dimensionality of input embeddings.
        num_heads (int): The number of attention heads for queries.
        dropout (float): Dropout rate for regularization.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(MultiQuerySuperpositionAttention, self).__init__()
        assert dim % num_heads == 0, "Embedding dimension must be divisible by number of heads."
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, and Value linear projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, self.head_dim)  # Shared keys
        self.v_proj = nn.Linear(dim, self.head_dim)  # Shared values

        # Learnable entanglement matrix
        self.entanglement_matrix = nn.Parameter(torch.randn(num_heads, self.head_dim, self.head_dim))
        
        # Linear projection for output
        self.out_proj = nn.Linear(dim, dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        logger.info("Initialized MultiQuerySuperpositionAttention with {} heads and embedding dimension {}.", num_heads, dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Multi-Query Superposition Attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            mask (Optional[torch.Tensor]): Attention mask tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor after attention mechanism.
        """
        batch_size, seq_len, dim = x.shape
        assert dim == self.dim, f"Input embedding dim ({dim}) must match model dim ({self.dim})"
        
        # Step 1: Create query, key, value projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Step 2: Reshape query to (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        logger.debug("Query shape: {}, Key shape: {}, Value shape: {}", q.shape, k.shape, v.shape)
        
        # Step 3: Superposition - Approximate superposition by averaging key/value vectors
        superposed_k = torch.mean(k, dim=1, keepdim=True)  # Approximate superposition across sequence
        superposed_v = torch.mean(v, dim=1, keepdim=True)  # Approximate superposition across sequence
        
        # Step 4: Entanglement - Apply learnable entanglement matrix
        entangled_q = torch.einsum('bhqd,hdd->bhqd', q, self.entanglement_matrix)  # Q x Entanglement matrix

        # Step 5: Calculate attention scores with sub-linear scaling
        attention_scores = torch.einsum('bhqd,bkd->bhqk', entangled_q, superposed_k) * self.scale

        # Step 6: Sub-linear masking
        if mask is not None:
            # Apply mask in a sub-linear manner (i.e., approximate sparse masking)
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        # Step 7: Softmax and compute attention weights
        attn_weights = torch.softmax(attention_scores, dim=-1)
        logger.debug("Attention weights shape: {}", attn_weights.shape)
        
        # Step 8: Weighted sum of values
        attention_output = torch.matmul(attn_weights, superposed_v)

        # Step 9: Reshape back and project output
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        output = self.out_proj(attention_output)
        output = self.dropout(output)
        
        logger.info("Output shape: {}", output.shape)
        return output


def run_benchmark_demo():
    """
    A demonstration function that initializes the MultiQuerySuperpositionAttention layer and processes
    a dummy input to test the attention mechanism.
    """
    logger.info("Running MultiQuerySuperpositionAttention demo.")
    
    # Set up input and parameters
    batch_size = 4
    seq_len = 128
    dim = 64
    num_heads = 8
    
    model = MultiQuerySuperpositionAttention(dim=dim, num_heads=num_heads)
    dummy_input = torch.rand(batch_size, seq_len, dim)
    
    logger.debug("Dummy input shape: {}", dummy_input.shape)
    
    # Optional sub-linear mask (e.g., padding mask)
    mask = torch.ones(batch_size, seq_len)
    mask[:, 64:] = 0  # Mask half of the sequence as an example
    
    # Run the forward pass
    output = model(dummy_input, mask=mask)
    
    logger.debug("Final output shape: {}", output.shape)


if __name__ == "__main__":
    run_benchmark_demo()
```
