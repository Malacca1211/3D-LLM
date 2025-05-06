import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, num_heads: int = 8, num_layers: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, embedding_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入形状: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # 投影到隐藏维度
        x = self.input_proj(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 投影到嵌入维度
        x = self.output_proj(x)
        
        return x 

class BaseEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 6
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

