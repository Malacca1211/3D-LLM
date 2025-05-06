import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from model.encoder import TransformerEncoder
from model.decoder import TransformerDecoder

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 初始化嵌入向量
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 将输入展平
        z_flattened = z.reshape(-1, self.embedding_dim)
        
        # 计算与码本的距离
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
            
        # 找到最近的嵌入向量
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # 计算损失
        loss = torch.mean((z_q.detach() - z) ** 2) + self.commitment_cost * torch.mean((z_q - z.detach()) ** 2)
        
        # 计算困惑度
        avg_probs = torch.mean(F.one_hot(min_encoding_indices, self.num_embeddings).float(), dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # 添加直通估计器
        z_q = z + (z_q - z).detach()
        
        return loss, z_q, perplexity

class VQVAE(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        input_dim: int = 6,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6
    ):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        self.decoder = TransformerDecoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        
    def forward(self, batch: dict) -> torch.Tensor:
        # 从字典中获取sketch_matrix
        x = batch['face_matrix']

        # print(f"x[0] shape: {x[0].shape}")
        # print(f"x[0] type: {type(x[0])}")
        # print(f"x[0]: {x[0]}")
        
        # 编码
        z = self.encoder(x)
        
        # 向量量化
        vq_loss, z_q, perplexity = self.vq_layer(z)
        
        # 解码
        x_recon = self.decoder(z_q)
        
        # 计算重构损失
        recon_loss = F.mse_loss(x_recon, x)
        
        # 总损失 = 重构损失 + VQ损失
        total_loss = recon_loss + vq_loss
        
        return total_loss, perplexity 