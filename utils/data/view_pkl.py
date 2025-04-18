import pickle
import random
import numpy as np

def print_random_samples(pkl_file_path, num_cases=5):
    try:
        # 读取 PKL 文件
        with open(pkl_file_path, 'rb') as file:
            data = pickle.load(file)
            
        # 获取所有样本
        total_cases = len(data)
        if total_cases < num_cases:
            print(f"警告：文件中只包含 {total_cases} 个样本，将全部显示")
            num_cases = total_cases
            
        # 随机选择样本
        selected_indices = random.sample(range(total_cases), num_cases)
        
        # 打印选中的样本
        for i, idx in enumerate(selected_indices, 1):
            sample = data[idx]
            print(f"\n{'='*50}")
            print(f"样本 {i}/{num_cases}:")
            print(f"{'='*50}")
            
            # 打印基本信息
            print(f"name: {sample['name']}")
            print(f"len_xy: {sample['len_xy']}") # num of points
            print(f"len_ext: {sample['len_ext']}") # num of extrude params
            print(f"len_pix: {sample['len_pix']}") # num of pixels
            print(f"len_cmd: {sample['len_cmd']}") # num of commands
            print(f"num_se: {sample['num_se']}") # num of steps
            print(f"se_xy: {sample['se_xy']}, shape: {np.array(sample['se_xy']).shape}")
            print(f"se_cmd: {sample['se_cmd']}, shape: {np.array(sample['se_cmd']).shape}")  # 命令代码: 3=线段命令(直线), 2=结束当前面, 1=结束当前环, 0=结束整个草图
            print(f"se_pix: {sample['se_pix']}, shape: {np.array(sample['se_pix']).shape}")
            print(f"se_ext: {sample['se_ext']}, shape: {np.array(sample['se_ext']).shape}")
            

            
            print(f"\n{'-'*50}")
            
    except FileNotFoundError:
        print("错误：找不到指定的文件")
    except pickle.UnpicklingError:
        print("错误：无法解析 PKL 文件，文件可能已损坏或格式不正确")
    except Exception as e:
        print(f"发生错误: {str(e)}")

# 设置要读取的 PKL 文件路径
pkl_file_path = '/home/malacca/3DLLM/data/skengen_data/train.pkl'

# 调用函数打印随机样本
print_random_samples(pkl_file_path)
