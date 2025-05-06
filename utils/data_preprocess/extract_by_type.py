import numpy as np
import pickle
import os
from pathlib import Path

def extract_face_matrix(pkl_file):
    """
    将sketch_matrix转换为标准化的numpy矩阵格式
    
    Args:
        sketch_matrix: 包含多个face的列表数据
        
    Returns:
        list of np.ndarray: 每个face对应的标准化矩阵
    """
    processed_matrices = []
    over_32_count = 0
    
    for i in range(len(pkl_file)):
        sketch_matrix = pkl_file[i]['sketch_matrix']
    
        for face in sketch_matrix:
            
            current_row = 0
            output_matrix = np.full((57, 8), -1, dtype=np.float32)
            

            for loop in face:
                edges = loop[:-1]
                is_outer = loop[-1]

                for edge in edges:
                    flattened_edge = [x for sublist in edge for x in sublist]
                    output_matrix[current_row, :] = flattened_edge
                    current_row += 1

                assert current_row < 57, f"current_row: {current_row}"
                
                output_matrix[current_row, :] = is_outer[0]  # 使用标识位填充整行
                current_row += 1
                
        if current_row > 32:
            over_32_count += 1
            print(f'over_32_count: {over_32_count}')
            continue
                    
    
        processed_matrices.append(output_matrix)
    
    return processed_matrices

def process_single_file(input_path, output_dir):
    """
    处理单个pkl文件
    
    Args:
        input_path: 输入pkl文件路径
        output_dir: 输出目录
    """
    # 读取输入文件
    with open(input_path, 'rb') as f:
        pkl_file = pickle.load(f)
    
    # 提取矩阵
    processed_matrices = extract_face_matrix(pkl_file)
    
    output_path = os.path.join(output_dir, f'validation_face.pkl')
    
    # 保存处理后的矩阵
    with open(output_path, 'wb') as f:
        pickle.dump(processed_matrices, f)
    
    print(f"已处理: {input_path} -> {output_path}")
    return output_path

def find_max_edges(input_path):
    """
    找到所有face中最大的edge数量,包括标记，最大为57
    """
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
        
    max_edges = 0
    
    print(f'len(data): {len(data)}')
    
    for i in range(len(data)):
        sketch_data = data[i]
        total_edges = 0
        for face in sketch_data['sketch_matrix']:
            for loop in face:
                total_edges += len(loop)
        max_edges = max(max_edges, total_edges)
    
    return max_edges

def stat_edge_distribution(input_path):
    """
    统计edge边的分布情况
    
    Args:
        input_path: 输入pkl文件路径
        
    Returns:
        dict: 包含edge数量分布统计的字典
    """
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    edge_distribution = {}
    
    for i in range(len(data)):
        sketch_data = data[i]
        edge_count = 0
        for face in sketch_data['sketch_matrix']:
            # edge_count = 0
            for loop in face:
                edge_count += len(loop)  # 不计算最后的is_outer标记
            
        if edge_count in edge_distribution:
            edge_distribution[edge_count] += 1
        else:
            edge_distribution[edge_count] = 1
    
    # 按edge数量排序
    sorted_distribution = dict(sorted(edge_distribution.items()))
    
    # 打印统计结果
    print("\nEdge数量分布统计:")
    print("Edge数量\t出现次数\t百分比")
    total_faces = sum(sorted_distribution.values())
    for edge_count, count in sorted_distribution.items():
        percentage = (count / total_faces) * 100
        print(f"{edge_count}\t\t{count}\t\t{percentage:.2f}%")
    
    return sorted_distribution

if __name__ == '__main__':
    # 设置输入输出路径
    output_dir = '/data-6t/malacca/3DLLM/data/3dllm_data/face'       # 处理后的矩阵保存目录
    
    # 处理单个文件
    input_path = '/data-6t/malacca/3DLLM/data/3dllm_data/train_dedup_face.pkl'
    process_single_file(input_path, output_dir)
    # max_edges = find_max_edges(input_path)
    # print(f'max_edges: {max_edges}')
    
    # 统计edge分布
    edge_distribution = stat_edge_distribution(input_path)

