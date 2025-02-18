
import torch.nn as nn
import torch
from src.attention import CustomMultiheadAttention

class PositionalEncoding(nn.Module):
    """
    Implements positional encoding to inject sequence order information into input embeddings.
    """
    def __init__(self, embed_dim: int, max_len: int = 20000):
        """
        Args:
            embed_dim (int): Dimension of the embeddings.
            max_len (int): Maximum sequence length. Default is 20000.
        """
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(pe.unsqueeze(0), requires_grad=False)
    
    def forward(self, x):
        """
        Adds positional encoding to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).
        
        Returns:
            torch.Tensor: Positional encoding tensor added to input tensor.
        """
        return self.pe[:, :x.size(1), :]

class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier for sequence classification tasks.
    """
    def __init__(
            self, 
            num_expression_levels: int, 
            embed_dim: int, 
            num_classes: int, 
            num_layers: int, 
            fc_hidden_dim: int, 
            num_heads: int, 
            drop_prob: float = 0.0,
            use_custom_attention: bool = True):
        """
        Args:
            num_expression_levels (int): Number of discrete gene expression levels for embedding.
            embed_dim (int): Dimension of the embedding vectors.
            num_classes (int): Number of output classes for classification.
            num_layers (int): Number of transformer layers.
            fc_hidden_dim (int): Hidden dimension of the feed-forward network.
            num_heads (int): Number of attention heads.
            drop_prob (float): Dropout probability. Default is 0.0.
            use_custom_attention (bool): Whether to use a custom multihead attention mechanism. Default is True.
        """
        super().__init__()

        # gene expression embedding
        self.exp_embedding = nn.Embedding(num_expression_levels, embed_dim)

        # positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim)

        # class token (learnable parameter)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # Shape: (1, 1, embed_dim)

        # transformer layers
        self.layers = nn.ModuleList([TransformerLayer(embed_dim, fc_hidden_dim, num_heads, drop_prob=drop_prob, use_custom_attention=use_custom_attention) for _ in range(num_layers)])

        # linear classification layer
        self.linear = nn.Linear(embed_dim, num_classes)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer classifier.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        # embed gene expression
        x = self.exp_embedding(x)
        
        # prepend class token to the sequence
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)  # Expand the class token to match batch size
        x = torch.cat([class_token, x], dim=1)  # Concatenate class token to the input sequence

        # add positional encoding
        x += self.pos_encoder(x)
        
        # forward through transformer layers
        for layer in self.layers:
            x = layer(x)

        # classify (using the output of the class token)
        x = self.linear(x[:, 0, :])  # Only use the output corresponding to the class token (first token)
        #x = self.softmax(x)
        
        return x
    
class TransformerLayer(nn.Module):
    """
    Single Transformer layer with attention and feed-forward sublayers.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int, drop_prob: float = 0.0, use_custom_attention: bool = True):
        """
        Args:
            embed_dim (int): Dimension of the embeddings.
            hidden_dim (int): Dimension of the feed-forward hidden layer.
            num_heads (int): Number of attention heads.
            drop_prob (float): Dropout probability. Default is 0.0.
            use_custom_attention (bool): Whether to use a custom multihead attention mechanism. Default is True.
        """
        super().__init__()
        if use_custom_attention:
            self.attn = CustomMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        else:
            self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, embed_dim).
        """
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