import os
from pathlib import Path

def find_coord_range(input_dir):
    """查找所有OBJ文件中坐标的最大最小值"""
    input_path = Path(input_dir)
    
    # 初始化最大最小值及其对应的文件路径
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')
    min_x_file = ""
    max_x_file = ""
    min_y_file = ""
    max_y_file = ""
    
    # 统计没有任何点超出±1范围的OBJ文件数量
    valid_files_count = 0
    # 存储超出范围的文件所在的文件夹路径
    out_of_range_dirs = set()  # 使用set去重
    
    # 获取所有OBJ文件
    obj_files = list(input_path.glob('**/*.obj'))
    print(f"找到 {len(obj_files)} 个文件需要处理")
    
    for obj_file in obj_files:
        print(f"处理文件: {obj_file}")
        with open(obj_file, 'r') as f:
            lines = f.readlines()
        
        # 用于跟踪当前文件中的坐标是否都在±1范围内
        file_valid = True
        
        for line in lines:
            if line.startswith('v'):
                parts = line.strip().split()
                if len(parts) == 3:  # v qx qy格式
                    x = float(parts[1])
                    y = float(parts[2])
                    
                    # 检查是否超出±1范围
                    if abs(x) > 1.0 or abs(y) > 1.0:
                        file_valid = False
                    
                    # 更新最大最小值及其对应的文件路径
                    if x < min_x:
                        min_x = x
                        min_x_file = str(obj_file)
                    if x > max_x:
                        max_x = x
                        max_x_file = str(obj_file)
                    if y < min_y:
                        min_y = y
                        min_y_file = str(obj_file)
                    if y > max_y:
                        max_y = y
                        max_y_file = str(obj_file)
        
        # 如果当前文件的所有点都在±1范围内，增加计数
        if file_valid:
            valid_files_count += 1
        else:
            # 添加文件所在的文件夹路径
            out_of_range_dirs.add(str(obj_file.parent))
    
    # 将超出范围的文件所在的文件夹路径写入文件
    output_file = "/home/malacca/3DLLM/data/out_of_range_dirs.txt"
    with open(output_file, 'w') as f:
        for dir_path in sorted(out_of_range_dirs):  # 排序以便于查看
            f.write(f"{dir_path}\n")
    
    print("\n坐标范围统计:")
    print(f"X坐标范围: [{min_x}, {max_x}]")
    print(f"最小X坐标文件: {min_x_file}")
    print(f"最大X坐标文件: {max_x_file}")
    print(f"Y坐标范围: [{min_y}, {max_y}]")
    print(f"最小Y坐标文件: {min_y_file}")
    print(f"最大Y坐标文件: {max_y_file}")
    print(f"\n没有任何点超出±1范围的OBJ文件数量: {valid_files_count}")
    print(f"找到 {len(obj_files)} 个文件需要处理")
    print(f"占比: {valid_files_count/len(obj_files)*100:.2f}%")
    print(f"\n超出范围的文件所在的文件夹路径已写入: {output_file}")
    print(f"共有 {len(out_of_range_dirs)} 个文件夹包含超出范围的文件")
    
    return min_x, max_x, min_y, max_y

if __name__ == "__main__":
    input_dir = "/home/malacca/3DLLM/data/cad_data_normalized"  # 输入目录
    min_x, max_x, min_y, max_y = find_coord_range(input_dir) 