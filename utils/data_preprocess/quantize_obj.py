import os
import numpy as np
from pathlib import Path
from tqdm import tqdm

def quantize_coordinate(x, min_val=-1.0, max_val=1.0):
    """将坐标值量化为8位整数"""
    # 将坐标值归一化到[0, 255]范围
    normalized = (x - min_val) / (max_val - min_val)
    quantized = int(normalized * 255)
    # 确保在有效范围内
    quantized = max(0, min(255, quantized))
    return quantized

def dequantize_coordinate(q, min_val=-1.0, max_val=1.0):
    """将8位整数反量化为原始坐标值"""
    normalized = q / 255.0
    return min_val + normalized * (max_val - min_val)

def process_obj_file(input_path, output_path):
    """处理单个OBJ文件"""
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 收集所有顶点数据
    vertices = []
    vertex_positions = []  # 记录原始顶点位置
    other_lines = []
    orig_vertex_count = 0
    
    for i, line in enumerate(lines):
        if line.startswith('v'):
            # 处理顶点行
            parts = line.strip().split()
            if len(parts) == 3:  # 确保是v qx qy格式
                orig_vertex_count += 1
                x = float(parts[1])
                y = float(parts[2])
                # 量化坐标
                qx = quantize_coordinate(x)
                qy = quantize_coordinate(y)
                vertices.append(f'v {qx} {qy}\n')
                vertex_positions.append(i)  # 记录顶点在原文件中的位置
            else:
                print("Not v qx qy format")
        else:
            # 保存其他行
            other_lines.append(line)
    
    # 写入新文件
    with open(output_path, 'w') as f:
        current_vertex = 0
        for i in range(len(lines)):
            if i in vertex_positions:
                # 在原始顶点位置写入量化后的顶点
                f.write(vertices[current_vertex])
                current_vertex += 1
            else:
                # 写入其他行
                f.write(lines[i])
            
    # 检查点数量是否一致
    if orig_vertex_count != len(vertices):
        print(f"警告: {input_path} 的点数量不一致!")
        print(f"原始点数: {orig_vertex_count}, 量化后点数: {len(vertices)}")

def process_directory(input_dir, output_dir):
    """处理整个目录下的OBJ文件"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 确保输出目录存在
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有OBJ文件并只处理前10个
    obj_files = list(input_path.glob('**/*.obj'))[:10]
    print(f"找到 {len(obj_files)} 个文件需要处理")
    
    for obj_file in tqdm(obj_files, desc="处理文件"):
        relative_path = obj_file.relative_to(input_path)
        output_file = output_path / relative_path
        process_obj_file(str(obj_file), str(output_file))

if __name__ == "__main__":
    input_dir = "/home/malacca/3DLLM/data/cad_data_normalized"  # 输入目录
    output_dir = "/home/malacca/3DLLM/data/cad_data_quantized"  # 输出目录
    
    process_directory(input_dir, output_dir)
    print("前10个文件量化处理完成！")