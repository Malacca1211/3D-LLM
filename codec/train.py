import os
import yaml

with open('codec/config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    gpu_ids = config['training'].get('gpu_ids', [0])
    gpu_ids_str = ','.join(map(str, gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from tqdm import tqdm
from model.trainer import VQVAETrainer

from model.vqvae import VQVAE
from utils.dataset import create_data_loaders

def main():
    # 初始化训练器
    trainer = VQVAETrainer('codec/config.yaml')
    
    # 开始训练
    trainer.train()
    
    # 在训练完成后进行测试
    # trainer.test()

if __name__ == '__main__':
    main() 