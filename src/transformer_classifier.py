
import torch.nn as nn
import torch

class TransformerClassifier(nn.Module):
    def __init__(self, num_expression_levels: int, embed_dim: int, num_classes: int, num_layers: int, fc_hidden_dim: int, num_heads: int):
        super().__init__()

        # gene expression embedding
        self.exp_embedding = nn.Embedding(num_expression_levels, embed_dim)

        # class token (learnable parameter)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Shape: (1, 1, embed_dim)

        # transformer layers
        self.layers = nn.ModuleList([TransformerLayer(embed_dim, fc_hidden_dim, num_heads) for _ in range(num_layers)])

        # linear classification layer
        self.linear = nn.Linear(embed_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        # embed gene expression
        x = self.exp_embedding(x)
        
        # prepend class token to the sequence
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)  # Expand the class token to match batch size
        x = torch.cat([class_token, x], dim=1)  # Concatenate class token to the input sequence
        
        # forward through transformer layers
        for layer in self.layers:
            x = layer(x)

        # classify (using the output of the class token)
        x = self.linear(x[:, 0, :])  # Only use the output corresponding to the class token (first token)
        x = self.softmax(x)
        
        return x
    
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int, drop_prob: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):

        # compute attention
        x_res, _ = self.attn(x, x, x)

        # add and norm
        x = self.norm1(x + x_res)

        # feed-forward nn
        x_res = self.linear1(x)
        x_res = self.relu(x_res)
        x_res = self.dropout(x_res)
        x_res = self.linear2(x_res)

        # add and norm
        x = self.norm2(x + x_res)

        return x