#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pickle
import json
from pathlib import Path
from tqdm import tqdm

class OBJMatrixConverter:
    def __init__(self, input_dir: str, output_dir: str, max_files: int = None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_edges = 4  # 每条边用4个点表示
        self.padding_value = -1  # 不足的用-1补齐
        self.max_files = max_files  # 最大处理文件数

    def parse_vertex(self, line: str) -> tuple:
        """解析顶点行，返回顶点坐标(x, y)"""
        parts = line.strip().split()
        if len(parts) >= 3:
            return (float(parts[1]), float(parts[2]))  # 直接返回x, y坐标
        return None

    def parse_edge(self, line: str) -> list:
        """解析边行，返回顶点索引列表"""
        parts = line.strip().split()
        if len(parts) >= 3:
            return [int(x) for x in parts[1:]]
        return None

    def process_face(self, vertices: list, edges: list) -> list:
        face_edges = []
        
        for edge in edges:
            # 获取当前边的所有点
            edge_points = []
            for v_idx in edge:
                if 0 <= v_idx < len(vertices):
                    edge_points.append(vertices[v_idx])
            
            # 确保每个边有4个点，不足的用(-1,-1)补充
            while len(edge_points) < 4:
                edge_points.append((-1, -1))
            
            # 只取前4个点
            edge_points = edge_points[:4]
            
            # 转换为嵌套列表格式
            edge_matrix = [list(point) for point in edge_points]
            face_edges.append(edge_matrix)
        
        # print(f"face_edges: {face_edges}")
        return face_edges

    def process_file(self, file_path: str) -> dict:
        """处理单个OBJ文件"""
        vertices = []  # 存储顶点坐标，索引即为顶点编号
        faces = []     # 存储每个face的矩阵
        current_face_edges = []  # 当前face的边
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith('v '):
                    vertex = self.parse_vertex(line)
                    if vertex:
                        vertices.append(vertex)
                
                elif line == 'face':
                    if current_face_edges:
                        face_matrix = self.process_face(vertices, current_face_edges)
                        faces.append(face_matrix)
                    current_face_edges = []
                
                elif line.startswith(('l', 'a', 'c')):
                    edge = self.parse_edge(line)
                    if edge:
                        current_face_edges.append(edge)
        
        # 处理最后一个face
        if current_face_edges:
            face_matrix = self.process_face(vertices, current_face_edges)
            faces.append(face_matrix)
        
        # 获取文件ID
        file_id = Path(file_path).stem
        if file_id.endswith('_param'):
            file_id = file_id[:-6]
        
        # 0086/00866023_000 format
        file_id = f"{file_id[:4]}/{file_id}"
        
        return {
            'file_id': file_id,
            'face_num': len(faces),
            'face_matrix': faces
        }

    def load_split_info(self, split_file: str) -> dict:
        """加载训练测试验证集划分信息"""
        with open(split_file, 'r') as f:
            return json.load(f)

    def process_all_files(self):
        """处理所有文件并根据划分保存结果"""
        # 加载划分信息
        split_file = '/home/malacca/3DLLM/data/train_val_test_split.json'
        split_info = self.load_split_info(split_file)
        
        # 初始化结果字典 - 使用统一的键名
        results = {
            'train': [],
            'validation': [],  # 改为validation
            'test': []
        }
        
        # 获取所有OBJ文件
        obj_files = list(Path(self.input_dir).glob('**/*.obj'))
        total_files = len(obj_files)
        print(f"找到 {total_files} 个OBJ文件需要处理")
        
        # 如果设置了最大处理文件数，则限制处理数量
        if self.max_files is not None:
            obj_files = obj_files[:self.max_files]
            print(f"将只处理前 {self.max_files} 个文件")
        
        for obj_file in tqdm(obj_files, desc="处理文件"):
            
            # print(f'split_info: {split_info["validation"][0]}')
            
            try:
                # 获取文件ID 和train_val_test.json中id格式匹配
                file_id = obj_file.stem
                if file_id.endswith('_param'):
                    file_id = file_id[:-10]
                file_id = f"{file_id[:4]}/{file_id}"

                
                # 处理文件
                file_data = self.process_file(str(obj_file))
                
                # print(f'file_data: {file_data["file_id"]}')
                # print(f'file_id: {file_id}')
                
                # 根据文件ID判断属于哪个数据集
                if file_id in split_info['train']:
                    results['train'].append(file_data)
                elif file_id in split_info['validation']:
                    results['validation'].append(file_data)
                elif file_id in split_info['test']:
                    results['test'].append(file_data)
                
            except Exception as e:
                print(f"处理文件 {obj_file} 时出错: {e}")
        
        # 保存结果时使用相同的键名
        split_mapping = {
            'train': 'train',
            'validation': 'val',  # 映射validation到val
            'test': 'test'
        }
        
        # 保存结果到不同的pkl文件
        for split_name, save_name in split_mapping.items():
            
            output_path = os.path.join(self.output_dir, f'sketch_{save_name}.pkl')
            with open(output_path, 'wb') as f:
                pickle.dump(results[split_name], f)
            print(f"\n{split_name}集处理完成，结果已保存到 {output_path}")
            print(f"共处理了 {len(results[split_name])} 个文件")
            
            # 打印统计信息
            total_faces = sum(data['face_num'] for data in results[split_name])
            print(f"总face数量: {total_faces}")
            print(f"平均每个文件的face数量: {total_faces/len(results[split_name]):.2f}")

def main():
    input_dir = '/home/malacca/3DLLM/data/cad_data_quantized'
    output_dir = '/home/malacca/3DLLM/data'
    
    # 先统计总文件数
    total_files = len(list(Path(input_dir).glob('**/*.obj')))
    print(f"目录中共有 {total_files} 个OBJ文件")
    
    # 处理文件
    converter = OBJMatrixConverter(input_dir, output_dir, max_files=200)
    converter.process_all_files()

if __name__ == "__main__":
    main() 