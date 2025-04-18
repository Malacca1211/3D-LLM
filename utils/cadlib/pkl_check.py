import os
import pickle
import numpy as np
from tqdm import tqdm

def load_pkl(file_path):
    """加载pickle文件并返回内容"""
    print(f"正在加载文件: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def analyze_pkl(data):
    """分析pickle数据的结构和内容"""
    print(f"\n数据类型: {type(data)}")
    
    if isinstance(data, list):
        print(f"列表长度: {len(data)}")
        if len(data) > 0:
            print("\n第一条数据的类型:", type(data[0]))
            if isinstance(data[0], dict):
                print("\n第一条数据的键:")
                for key in data[0].keys():
                    value = data[0][key]
                    value_type = type(value)
                    value_info = ""
                    if isinstance(value, np.ndarray):
                        value_info = f", 形状: {value.shape}, 数据类型: {value.dtype}"
                    elif isinstance(value, list):
                        value_info = f", 长度: {len(value)}"
                        if len(value) > 0:
                            value_info += f", 元素类型: {type(value[0])}"
                    print(f"  - {key}: {value_type}{value_info}")
    elif isinstance(data, dict):
        print(f"字典键的数量: {len(data.keys())}")
        print("\n字典的键:")
        for key in data.keys():
            value = data[key]
            value_type = type(value)
            value_info = ""
            if isinstance(value, np.ndarray):
                value_info = f", 形状: {value.shape}, 数据类型: {value.dtype}"
            elif isinstance(value, list):
                value_info = f", 长度: {len(value)}"
            print(f"  - {key}: {value_type}{value_info}")

def sample_data(data, num_samples=5):
    """从数据中随机抽取并展示几个样本"""
    if isinstance(data, list) and len(data) > 0:
        print(f"\n随机抽取 {num_samples} 个样本进行展示:")
        indices = np.random.choice(len(data), min(num_samples, len(data)), replace=False)
        for i, idx in enumerate(indices):
            print(f"\n样本 {i+1} (索引 {idx}):")
            if isinstance(data[idx], dict):
                for key, value in data[idx].items():
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    print(f"  {key}: {value_str}")
            else:
                print(f"  {data[idx]}")

def examine_values(data, key, num_samples=5):
    """检查特定键的值分布"""
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        if key in data[0]:
            print(f"\n检查键 '{key}' 的值:")
            values = []
            for item in data:
                if key in item:
                    values.append(item[key])
            
            if isinstance(values[0], (int, float, np.number)):
                values = np.array(values)
                print(f"  最小值: {np.min(values)}")
                print(f"  最大值: {np.max(values)}")
                print(f"  平均值: {np.mean(values)}")
                print(f"  中位数: {np.median(values)}")
            elif isinstance(values[0], (list, np.ndarray)):
                lengths = [len(val) for val in values]
                print(f"  长度最小值: {min(lengths)}")
                print(f"  长度最大值: {max(lengths)}")
                print(f"  长度平均值: {np.mean(lengths)}")
                print(f"  长度中位数: {np.median(lengths)}")
            
            # 随机抽样展示
            indices = np.random.choice(len(values), min(num_samples, len(values)), replace=False)
            print(f"\n  随机 {num_samples} 个 '{key}' 值样本:")
            for i, idx in enumerate(indices):
                value_str = str(values[idx])
                if len(value_str) > 100:
                    value_str = value_str[:100] + "..."
                print(f"    样本 {i+1}: {value_str}")
        else:
            print(f"键 '{key}' 在数据中不存在")

def main():
    # 使用变量设置参数，而不是命令行传入
    file_path = "/home/malacca/3DLLM/data/cad_data/train_deduplicate_e.pkl"  # 要分析的pickle文件路径
    key_to_check = None  # 要特别检查的键名，设为None则不检查特定键
    num_samples = 5  # 要显示的样本数量
    
    # 加载pickle文件
    data = load_pkl(file_path)
    
    # 分析数据结构
    analyze_pkl(data)
    
    # 显示样本数据
    sample_data(data, num_samples)
    
    # 如果指定了特定键，分析该键的值
    if key_to_check:
        examine_values(data, key_to_check, num_samples)

if __name__ == "__main__":
    main()
