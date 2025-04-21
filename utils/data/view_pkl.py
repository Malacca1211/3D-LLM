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

def view_pkl(pkl_path: str, max_samples: int = 100):
    """查看pkl文件的内容，适配新的face matrix格式
    
    Args:
        pkl_path: pkl文件路径
        max_samples: 最多显示的样本数
    """
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"总文件数: {len(data)}")
        print("\n" + "="*50)
        
        # 显示前max_samples个样本
        for i, file_data in enumerate(data):
            if i >= max_samples:
                break
                
            print(f"\n文件 {i+1}:")
            print(f"file_id: {file_data['file_id']}")
            print(f"face_num: {file_data['face_num']}")
            
            for i in file_data:
                print(i)
            
            # 显示每个face的loops和edges
            for face_idx, face in enumerate(file_data['sketch_matrix']):
                print(f"\n面 {face_idx + 1}:")
                
                # 遍历每个loop
                for loop_idx, loop in enumerate(face):
                    # 最后一个元素是in/out标记
                    is_outer = loop[-1] == [1, 1]
                    loop_type = "外环" if is_outer else "内环"
                    print(f"  Loop {loop_idx + 1} ({loop_type}):")
                    
                    # 遍历loop中的每条边（除了最后一个in/out标记）
                    for edge_idx, edge in enumerate(loop[:-1]):
                        print(f"    边 {edge_idx + 1}:")
                        for point_idx, point in enumerate(edge):
                            # if point != [-1, -1]:  # 只显示非填充点
                            print(f"      点 {point_idx + 1}: [{point[0]:.2f}, {point[1]:.2f}]")
            
            print("\n" + "-"*50)
            
    except Exception as e:
        print(f"读取pkl文件时出错: {e}")

if __name__ == "__main__":
    pkl_file_path = '/home/malacca/3DLLM/data/sketch_train.pkl'  # 更新文件路径
    
    print("\n=== 文件结构信息 ===")
    view_pkl(pkl_file_path)
