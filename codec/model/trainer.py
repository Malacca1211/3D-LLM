import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from .vqvae import VQVAE
from utils.dataset import create_data_loaders

class VQVAETrainer:
    def __init__(self, config_path: str):
        # 加载配置文件
        self.config = self._load_config(config_path)
        
        # GPU设置
        self._setup_device()
        
        self.train_mode = self.config['training']['train_mode']
        self.model_config = self.config['model'][self.train_mode]
        
        # 初始化模型
        self.model = self._init_model()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=float(self.config['training']['learning_rate'])
        )
        
        # 初始化TensorBoard
        self._init_tensorboard()

    def _setup_device(self):
        """设置训练设备和GPU"""
        if self.config['training']['device'] == 'cuda':
            if torch.cuda.is_available():
                # 获取指定的GPU ID
                gpu_ids = self.config['training'].get('gpu_ids', [0])
                
                # 设置可见的GPU
                gpu_ids_str = ','.join(map(str, gpu_ids))
                # os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str 此处无效，必须在import torch之前设置
                
                # 设置主GPU
                self.device = torch.device(f'cuda:{0}')  # 使用第一个可见的GPU
                
                # 打印GPU信息
                print(f"使用GPU: {gpu_ids_str}")
                print(f"当前主GPU: {torch.cuda.get_device_name(0)}, gpu_ids: {gpu_ids[0]}")
                
                # 如果有多个GPU，设置并行
                if len(gpu_ids) > 1:
                    self.use_multi_gpu = True
                else:
                    self.use_multi_gpu = False
            else:
                print("警告：配置指定使用CUDA，但未检测到可用的GPU。将使用CPU进行训练。")
                self.device = torch.device('cpu')
                self.use_multi_gpu = False
        else:
            print("使用CPU进行训练")
            self.device = torch.device('cpu')
            self.use_multi_gpu = False

    def _init_model(self) -> VQVAE:
        model = VQVAE(
            num_embeddings=self.model_config['num_embeddings'],
            embedding_dim=self.model_config['embedding_dim'],
            commitment_cost=self.model_config['commitment_cost'],
            input_dim=self.model_config['input_dim'],
            hidden_dim=self.model_config['hidden_dim']
        ).to(self.device)
        
        # 如果使用多GPU，则启用数据并行
        if self.use_multi_gpu:
            model = nn.DataParallel(model)
            print(f"启用多GPU并行训练，使用GPU数量: {torch.cuda.device_count()}")
            
        return model

    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _init_tensorboard(self):
        """初始化TensorBoard日志记录器"""
        log_dir = os.path.join(
            self.config['training']['log_dir'],
            datetime.now().strftime('%Y%m%d-%H%M%S')
        )
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard日志保存在: {log_dir}")

    def train(self):
        train_dataloader = create_data_loaders(
            data_path=self.config['data']['train'][self.train_mode],
            mode=self.train_mode,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers']
        )

        for epoch in range(self.config['training']['num_epochs']):
            self.model.train()
            total_loss = 0
            total_perplexity = 0
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch+1}')):
                loss, perplexity = self._train_step(batch)
                total_loss += loss
                total_perplexity += perplexity
                
                # 记录每个batch的损失
                self.writer.add_scalar('Loss/train_batch', loss, epoch * len(train_dataloader) + batch_idx)
            
            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch+1}, Average Training Loss: {avg_loss:.4f}')
            # 记录每个epoch的平均损失
            self.writer.add_scalar('Loss/train_epoch', avg_loss, epoch)
            
            avg_perplexity = total_perplexity / len(train_dataloader)
            print(f'Epoch {epoch+1}, Average Training Perplexity: {avg_perplexity:.4f}')
            # 记录每个epoch的平均困惑度
            self.writer.add_scalar('Perplexity/train_epoch', avg_perplexity, epoch)

            # 进行验证
            val_loss = self.validate()
            self.writer.add_scalar('Loss/val_epoch', val_loss, epoch)
            
            # 保存模型检查点
            if epoch % self.config['training']['save_every'] == 0:
                self.save_model(epoch)
            
        # 关闭TensorBoard写入器
        self.writer.close()

    def _train_step(self, batch) -> float:
        # 将字典中的每个张量移动到指定设备
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        self.optimizer.zero_grad()
        loss, perplexity = self.model(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item(), perplexity
        
    def _validation_step(self, batch) -> float:
        # 将字典中的每个张量移动到指定设备
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        
        # 在验证阶段不需要计算梯度
        with torch.no_grad():
            loss, perplexity = self.model(batch)
        return loss.item()

    def validate(self) -> float:
        val_dataloader = create_data_loaders(
            data_path=self.config['data']['val'],
            mode=self.train_mode,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers']
        )

        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(val_dataloader, desc='Validating'):
            loss = self._validation_step(batch)
            total_loss += loss
        
        avg_loss = total_loss / len(val_dataloader)
        # print(f'Validation Loss: {avg_loss:.4f}')
        return avg_loss

    def test(self) -> float:
        test_dataloader = create_data_loaders(
            data_path=self.config['data']['test'],
            mode=self.train_mode,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['training']['num_workers']
        )

        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc='Testing'):
                batch = batch.to(self.device)
                loss = self.model(batch)
                total_loss += loss.item()
        
        test_loss = total_loss / len(test_dataloader)
        print(f'Test Loss: {test_loss:.4f}')
        return test_loss

    def save_model(self, epoch: int):
        # 创建保存目录（如果不存在）
        save_dir = self.config['training']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(
            save_dir,
            f'model_{self.train_mode}_epoch_{epoch}.pt'
        )
        
        # 如果是DataParallel模型，保存内部的模型
        model_to_save = self.model.module if self.use_multi_gpu else self.model
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

    def load_model(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 如果是DataParallel模型，加载到内部的模型
        if self.use_multi_gpu:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
