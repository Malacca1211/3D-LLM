import pickle
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Union, Any, Tuple
import numpy as np

class FaceDataset(Dataset):
    def __init__(self, data_path: str):
        
        self.data_path = data_path
        
        # 加载数据
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
            
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        face_matrix = self.data[idx]
        return {
            'face_matrix': face_matrix.astype(np.float32)
        }

class ExtrudeDataset(Dataset):
    def __init__(self, data_path: str):
        super().__init__(data_path)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        获取单个拉伸特征数据
        
        Returns:
            Dict包含拉伸特征的参数和属性
        """
        extrude_data = self.data[idx]
        return {
            'file_id': extrude_data['file_id'],
            'face_num': extrude_data['face_num'],
            'operation_type': extrude_data['operation_type'],
            'extrude_param': np.array(extrude_data['extrude_param']),
            'transform_matrix': np.array(extrude_data['transform_matrix'])
        }

def create_data_loaders(
    data_path: str,
    mode: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    
    if mode == 'face':
        dataset = FaceDataset(data_path)
    elif mode == 'loop':
        # dataset = SketchDataset(data_path)
        pass
    elif mode == 'extrude':
        dataset = ExtrudeDataset(data_path)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

if __name__ == '__main__':
    # 测试数据路径
    face_data_path = '/data-6t/malacca/3DLLM/data/3dllm_data/face/test_face.pkl'
    
    # 创建数据集
    dataloader = create_data_loaders(face_data_path, 'face')
    print(f'dataset size: {len(dataloader)}')
    
    # 测试数据加载
    print("\n测试数据加载:")
    batch = next(iter(dataloader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Sketch matrix shape: {batch['face_matrix'].shape}")
    print(f"Sketch matrix dtype: {batch['face_matrix'].dtype}")
    
    # 打印第一个样本的内容
    print("\n第一个样本的矩阵内容:")
    print(batch['face_matrix'][0])
    print(f"batch size: {len(batch['face_matrix'])}")
    

    
