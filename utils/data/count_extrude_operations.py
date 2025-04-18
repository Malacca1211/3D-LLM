#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from collections import Counter
import time

# 定义要扫描的目录
base_dir = '/home/malacca/3DLLM/data/cad_data_normalized'

# 统计变量
extrude_types = Counter()
files_without_extrude = 0
total_obj_files = 0
processed_files = 0

# 用于保存每种类型的示例文件路径
example_files = {}

# 定义标准的变换矩阵值
standard_transforms = {
    "T_origin": [0.0, 0.0, 0.0],
    "T_xaxis": [1.0, 0.0, 0.0],
    "T_yaxis": [0.0, 1.0, 0.0],
    "T_zaxis": [0.0, 0.0, 1.0]
}

# 非标准变换矩阵计数
non_standard_transforms = {
    "T_origin": 0,
    "T_xaxis": 0,
    "T_yaxis": 0,
    "T_zaxis": 0
}

# 非标准变换示例文件
non_standard_examples = {
    "T_origin": None,
    "T_xaxis": None,
    "T_yaxis": None,
    "T_zaxis": None
}

# 正则表达式模式，用于提取ExtrudeOperation的值
extrude_pattern = re.compile(r'ExtrudeOperation:\s*(\S+)')
# 正则表达式模式，用于提取变换矩阵值
transform_patterns = {
    "T_origin": re.compile(r'T_origin\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)'),
    "T_xaxis": re.compile(r'T_xaxis\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)'),
    "T_yaxis": re.compile(r'T_yaxis\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)'),
    "T_zaxis": re.compile(r'T_zaxis\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)')
}

# 获取总文件数用于显示进度
def count_total_files():
    file_count = 0
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.obj'):
                file_count += 1
    return file_count

estimated_total = count_total_files()
print(f"预计处理 {estimated_total} 个 OBJ 文件...")
start_time = time.time()

# 递归遍历所有目录
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.obj'):
            total_obj_files += 1
            file_path = os.path.join(root, file)
            
            # 显示进度
            processed_files += 1
            if processed_files % 1000 == 0 or processed_files == estimated_total:
                elapsed = time.time() - start_time
                percent = (processed_files / estimated_total) * 100 if estimated_total > 0 else 0
                print(f"进度: {processed_files}/{estimated_total} ({percent:.1f}%) - 已用时间: {elapsed:.1f}秒")
            
            try:
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 查找ExtrudeOperation
                match = extrude_pattern.search(content)
                if match:
                    operation_type = match.group(1)
                    extrude_types[operation_type] += 1
                    
                    # 保存每种类型的第一个示例文件
                    if operation_type not in example_files:
                        example_files[operation_type] = file_path
                else:
                    files_without_extrude += 1
                
                # 检查变换矩阵
                for key, pattern in transform_patterns.items():
                    match = pattern.search(content)
                    if match:
                        values = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
                        if values != standard_transforms[key]:
                            non_standard_transforms[key] += 1
                            if non_standard_examples[key] is None:
                                non_standard_examples[key] = file_path
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {e}")

# 打印结果
print("\n===== 挤出操作类型统计 =====")
print(f"总计扫描 OBJ 文件数: {total_obj_files}")
print("\n挤出操作类型及数量:")
for op_type, count in extrude_types.most_common():
    print(f"  {op_type}: {count} 个文件")
    print(f"    示例文件: {example_files[op_type]}")

print(f"\n不包含挤出操作的文件数: {files_without_extrude}")

# 计算包含挤出操作的文件百分比
files_with_extrude = total_obj_files - files_without_extrude
if total_obj_files > 0:
    percentage = (files_with_extrude / total_obj_files) * 100
    print(f"\n包含挤出操作的文件百分比: {percentage:.2f}%")

# 打印变换矩阵统计结果
print("\n===== 变换矩阵统计 =====")
print("标准变换矩阵值:")
for key, value in standard_transforms.items():
    print(f"  {key}: {value}")

print("\n非标准变换矩阵数量:")
for key, count in non_standard_transforms.items():
    print(f"  {key} 不为标准值的文件数: {count}")
    if non_standard_examples[key]:
        print(f"    示例文件: {non_standard_examples[key]}")

total_time = time.time() - start_time
print(f"\n总耗时: {total_time:.2f}秒") 