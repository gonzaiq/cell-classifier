import torch.nn as nn
import torch

class CustomMultiheadAttention(nn.Module):

    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()

        assert embed_dim % num_heads == 0, "The embedding dimension must be divisible by the number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Q, K, V projection matrices
        """
        self.q_proj = nn.Parameter(torch.empty((embed_dim, embed_dim)))
        self.k_proj = nn.Parameter(torch.empty((embed_dim, embed_dim)))
        self.v_proj = nn.Parameter(torch.empty((embed_dim, embed_dim)))
        """

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):

        batch_dim, seq_dim, embed_dim = query.shape

        assert self.embed_dim == embed_dim, f"Expected embedding dimension {self.embed_dim}, but got {embed_dim}"

        # get projected query, key, and value
        # shape (batch, seq, embed_dim), with embed_dim = head * head_emb
        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        # reshape for calculating multi-head attention (batch, head, seq, head_emb)
        query = query.view(batch_dim, seq_dim, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_dim, seq_dim, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_dim, seq_dim, self.num_heads, self.head_dim).transpose(1, 2)

        # compute scaled dot product between queries and key
        scaled_dot_product = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        att_weights = self.softmax(scaled_dot_product)  # shape (batch, head, seq, seq)

        # compute attention
        att_out = torch.matmul(att_weights, value) # shape (batch, head, seq, head_emb)

        # reshape and concatenate to get original embedding dim
        att_out = att_out.transpose(1, 2) # shape (batch, seq, head, head_emb)
        att_out = att_out.reshape(batch_dim, seq_dim, self.embed_dim) # shape (batch, seq, embed_dim)

        # apply linear layer
        att_out = self.out_linear(att_out)

        return att_out, att_weights

