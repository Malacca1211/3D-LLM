import numpy as np
from hashlib import sha256
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import os

NUM_THREADS = 36
EXTRA_PAD = 1

def hash_face(data, include_loop_type=True):
    """对整个face进行哈希"""
    if not data['sketch_matrix']:
        return '', ''
    
    face_data = []
    for face in data['sketch_matrix']:
        for loop in face:
            # 点坐标
            for edge in loop[:-1]:
                for point in edge:
                    face_data.extend([p + EXTRA_PAD for p in point])
            # 可选择是否包含内外环标记
            if include_loop_type:
                face_data.extend(loop[-1])
    
    face_array = np.array(face_data, dtype=np.float32)
    face_hash = sha256(face_array.tobytes()).hexdigest()
    return face_hash, data['file_id']

def hash_loop(data):
    """对每个loop单独进行哈希"""
    if not data['sketch_matrix']:
        return [], ''
    
    loop_hashes = []
    for face in data['sketch_matrix']:
        for loop in face:
            loop_points = []
            for edge in loop[:-1]:  # 除去标记
                for point in edge:
                    loop_points.extend([p + EXTRA_PAD for p in point])
            loop_array = np.array(loop_points, dtype=np.float32)
            loop_hash = sha256(loop_array.tobytes()).hexdigest()
            loop_hashes.append(loop_hash)
    
    return loop_hashes, data['file_id']

def hash_extrude(data):
    """对挤出操作相关参数进行哈希"""
    if not data['extrude_param'] or not data['transform_matrix']:
        return '', ''
    
    # 合并挤出参数和变换矩阵
    extrude_data = []
    # 添加挤出参数
    for param in data['extrude_param']:
        extrude_data.extend([p + EXTRA_PAD for p in param])
    
    # 添加变换矩阵
    for row in data['transform_matrix']:
        extrude_data.extend([p + EXTRA_PAD for p in row])
    
    extrude_array = np.array(extrude_data, dtype=np.float32)
    extrude_hash = sha256(extrude_array.tobytes()).hexdigest()
    return extrude_hash, data['file_id']

def parallel_hash_data(data_list, hash_type):
    """并行处理哈希计算"""
    hash_funcs = {
        'face': hash_face,
        'loop': hash_loop,
        'extrude': hash_extrude
    }
    
    hash_func = hash_funcs[hash_type]
    duplicate_groups = {}
    
    with Pool(NUM_THREADS) as pool:
        results = list(tqdm(
            pool.imap(hash_func, data_list),
            total=len(data_list),
            desc=f"Hashing {hash_type}"
        ))
    
    for hashes, file_id in results:
        if hash_type == 'loop':  # loop返回多个哈希值
            for h in hashes:
                if h not in duplicate_groups:
                    duplicate_groups[h] = []
                duplicate_groups[h].append(file_id)
        else:  # face和extrude返回单个哈希值
            h = hashes
            if h and len(h) > 0:
                if h not in duplicate_groups:
                    duplicate_groups[h] = []
                duplicate_groups[h].append(file_id)
    
    return duplicate_groups

def deduplicate_data(data_list, hash_type):
    """执行去重操作"""
    duplicate_groups = parallel_hash_data(data_list, hash_type)
    
    # 统计重复情况
    unique_files = set()
    for group in duplicate_groups.values():
        unique_files.add(group[0])  # 只保留每组第一个文件
    
    # 生成去重后的数据集
    deduped_data = [d for d in data_list if d['file_id'] in unique_files]
    
    # 计算统计信息
    total = len(data_list)
    unique = len(deduped_data)
    duplicate_rate = (total - unique) / total * 100
    
    stats = {
        'total_count': total,
        'unique_count': unique,
        'duplicate_rate': duplicate_rate
    }
    
    return deduped_data, stats

if __name__ == "__main__":
    # 设置输入输出路径
    data_dir = "/home/malacca/3DLLM/data/3dllm_data"
    hash_type = "extrude"  # 可选: face, loop, extrude
    
    # 加载数据
    print("正在加载数据...")
    with open(os.path.join(data_dir, "train.pkl"), "rb") as f:
        train_data = pickle.load(f)
    
    # 执行去重
    print(f"使用{hash_type}方式进行去重...")
    deduped_data, stats = deduplicate_data(train_data, hash_type)
    
    # 保存去重后的数据
    output_path = os.path.join(data_dir, f"train_dedup_{hash_type}.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(deduped_data, f)
    
    # 打印统计信息
    print("\n去重统计信息:")
    print(f"原始数据量: {stats['total_count']}")
    print(f"去重后数据量: {stats['unique_count']}")
    print(f"重复率: {stats['duplicate_rate']:.2f}%")
    print(f"去重后的数据已保存至: {output_path}")
