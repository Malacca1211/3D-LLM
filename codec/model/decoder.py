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

class BaseDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 6,
        num_heads: int = 8,
        num_layers: int = 6
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers


class TransformerDecoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 8, num_layers: int = 6):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 输入投影
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # 输入形状: (batch_size, seq_len, embedding_dim)
        batch_size = z.size(0)
        
        # 投影到隐藏维度
        z = self.input_proj(z)
        
        # 添加位置编码
        z = self.pos_encoder(z)
        
        # Transformer解码
        z = self.transformer_decoder(z, z)  # 自注意力
        
        # 投影到输出维度
        x_recon = self.output_proj(z)
        
        return x_recon 