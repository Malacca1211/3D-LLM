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
        # 添加操作类型映射字典
        self.operation_type_map = {
            'NewBodyFeatureOperation': 0,
            'JoinFeatureOperation': 1,
            'CutFeatureOperation': 2,
            'IntersectFeatureOperation': 3
        }

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

    def process_face(self, vertices: list, edges: list, is_outer: bool = True) -> list:
        """处理一个face的所有边
        Args:
            vertices: 顶点列表
            edges: 边列表
            is_outer: 是否是外环 (True表示out, False表示in)
        Returns:
            包含所有边和in/out标记的列表
        """
        edge_points = []
        
        # 处理所有边
        for edge in edges:
            # 获取当前边的所有点
            points = []
            for v_idx in edge:
                if 0 <= v_idx < len(vertices):
                    points.append(vertices[v_idx])
            
            # 确保每个边有4个点，不足的用(-1,-1)补充
            while len(points) < 4:
                points.append((-1, -1))
            
            # 只取前4个点
            points = points[:4]
            
            # 转换为嵌套列表格式
            edge_matrix = [list(point) for point in points]
            edge_points.append(edge_matrix)
        
        # 添加in/out标记
        edge_points.append([1, 1] if is_outer else [0, 0])
        
        return edge_points

    def parse_extrude_info(self, file_path: str) -> dict:
        """解析挤出相关信息"""
        extrude_info = {
            'operation_type': None,
            'extrude_param': None,
            'transform_matrix': np.zeros((4, 3))  # 4x3矩阵存储变换信息
        }
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # 解析操作类型，只取冒号后面的部分
                if 'ExtrudeOperation:' in line:
                    operation = line.split(':')[1].strip()
                    extrude_info['operation_type'] = self.operation_type_map[operation]
                
                # 解析挤出参数
                elif line.startswith('Extrude'):
                    parts = line.split()
                    if len(parts) == 3:
                        extrude_info['extrude_param'] = [[float(parts[1]), float(parts[2])]]
                
                # 解析变换矩阵
                elif line.startswith('T_origin'):
                    parts = line.split()
                    if len(parts) == 4:
                        extrude_info['transform_matrix'][0] = [float(parts[1]), float(parts[2]), float(parts[3])]
                elif line.startswith('T_xaxis'):
                    parts = line.split()
                    if len(parts) == 4:
                        extrude_info['transform_matrix'][1] = [float(parts[1]), float(parts[2]), float(parts[3])]
                elif line.startswith('T_yaxis'):
                    parts = line.split()
                    if len(parts) == 4:
                        extrude_info['transform_matrix'][2] = [float(parts[1]), float(parts[2]), float(parts[3])]
                elif line.startswith('T_zaxis'):
                    parts = line.split()
                    if len(parts) == 4:
                        extrude_info['transform_matrix'][3] = [float(parts[1]), float(parts[2]), float(parts[3])]
        
        return extrude_info

    def process_file(self, file_path: str) -> dict:
        """处理单个OBJ文件"""
        vertices = []  # 存储顶点坐标
        faces = []     # 存储每个face的矩阵
        current_face = None  # 当前face
        current_face_edges = []  # 当前face的边
        is_outer = True  # 当前是否是外环
        
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
                    # 如果有未处理的edges，先处理完
                    if current_face_edges and current_face is not None:
                        loop_matrix = self.process_face(vertices, current_face_edges, is_outer)
                        current_face.append(loop_matrix)
                    
                    # 如果有之前的face，添加到faces列表
                    if current_face is not None:
                        faces.append(current_face)
                    
                    # 开始新的face
                    current_face = []  # 新建face列表
                    current_face_edges = []  # 重置当前edges
                    is_outer = True  # 重置为外环
                    
                elif line == 'out':
                    # 如果当前有edges，先处理完
                    if current_face_edges and current_face is not None:
                        loop_matrix = self.process_face(vertices, current_face_edges, is_outer)
                        current_face.append(loop_matrix)
                    current_face_edges = []
                    is_outer = True
                    
                elif line == 'in':
                    # 如果当前有edges，先处理完
                    if current_face_edges and current_face is not None:
                        loop_matrix = self.process_face(vertices, current_face_edges, is_outer)
                        current_face.append(loop_matrix)
                    current_face_edges = []
                    is_outer = False
                    
                elif line.startswith(('l', 'a', 'c')):
                    edge = self.parse_edge(line)
                    if edge:
                        current_face_edges.append(edge)
        
            # 处理最后一组edges和face
            if current_face_edges and current_face is not None:
                loop_matrix = self.process_face(vertices, current_face_edges, is_outer)
                current_face.append(loop_matrix)
            if current_face is not None:
                faces.append(current_face)
        
        # 获取挤出相关信息
        extrude_info = self.parse_extrude_info(file_path)
        
        # 获取文件ID
        file_id = Path(file_path).stem
        if file_id.endswith('_param'):
            file_id = file_id[:-6]
        file_id = f"{file_id[:4]}/{file_id}"
        
        return {
            'file_id': file_id,
            'face_num': len(faces),
            'sketch_matrix': faces,  # 改名为sketch_matrix
            'operation_type': extrude_info['operation_type'],
            'extrude_param': extrude_info['extrude_param'],
            'transform_matrix': extrude_info['transform_matrix'].tolist()
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
            
            # try:
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
                
            # except Exception as e:
            #     print(f"处理文件 {obj_file} 时出错: {e}")
        
        # 保存结果时使用相同的键名
        split_mapping = {
            'train': 'train',
            'validation': 'validation',  # 映射validation到val
            'test': 'test'
        }
        
        # 保存结果到不同的pkl文件
        for split_name, save_name in split_mapping.items():
            
            output_path = os.path.join(self.output_dir, f'{save_name}.pkl')
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
    output_dir = '/home/malacca/3DLLM/data/3dllm_data'
    
    # 先统计总文件数
    total_files = len(list(Path(input_dir).glob('**/*.obj')))
    print(f"目录中共有 {total_files} 个OBJ文件")
    
    # 处理文件
    converter = OBJMatrixConverter(input_dir, output_dir, max_files=None)
    converter.process_all_files()

if __name__ == "__main__":
    main() 