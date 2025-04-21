#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from collections import Counter
import time
from typing import Dict, List, Tuple

class OBJFileAnalyzer:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.standard_transforms = {
            "T_origin": [0.0, 0.0, 0.0],
            "T_xaxis": [1.0, 0.0, 0.0],
            "T_yaxis": [0.0, 1.0, 0.0],
            "T_zaxis": [0.0, 0.0, 1.0]
        }
        self.patterns = {
            "extrude": re.compile(r'ExtrudeOperation:\s*(\S+)'),
            "face": re.compile(r'^face$', re.MULTILINE),
            "T_origin": re.compile(r'T_origin\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)'),
            "T_xaxis": re.compile(r'T_xaxis\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)'),
            "T_yaxis": re.compile(r'T_yaxis\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)'),
            "T_zaxis": re.compile(r'T_zaxis\s+([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)')
        }

    def count_total_files(self) -> int:
        """统计目录下所有OBJ文件的数量"""
        file_count = 0
        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.obj'):
                    file_count += 1
        return file_count

    def analyze_extrude_operations(self, file_path: str) -> Tuple[str, bool]:
        """分析文件中的挤出操作类型"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        match = self.patterns["extrude"].search(content)
        return (match.group(1) if match else None, match is not None)

    def count_faces(self, file_path: str) -> int:
        """统计文件中的face数量"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return len(self.patterns["face"].findall(content))

    def check_transforms(self, file_path: str) -> Dict[str, bool]:
        """检查变换矩阵是否标准"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        results = {}
        for key in ["T_origin", "T_xaxis", "T_yaxis", "T_zaxis"]:
            match = self.patterns[key].search(content)
            if match:
                values = [float(match.group(1)), float(match.group(2)), float(match.group(3))]
                results[key] = values != self.standard_transforms[key]
        return results

    def analyze_all_files(self, show_progress: bool = True) -> Dict:
        """分析所有文件并返回统计结果"""
        results = {
            "total_files": 0,
            "extrude_types": Counter(),
            "files_without_extrude": 0,
            "files_with_many_faces": 0,
            "non_standard_transforms": Counter(),
            "example_files": {},
            "many_faces_examples": [],
            "non_standard_examples": {}
        }

        total_files = self.count_total_files()
        if show_progress:
            print(f"预计处理 {total_files} 个 OBJ 文件...")
        start_time = time.time()

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.obj'):
                    results["total_files"] += 1
                    file_path = os.path.join(root, file)

                    if show_progress and results["total_files"] % 1000 == 0:
                        elapsed = time.time() - start_time
                        percent = (results["total_files"] / total_files) * 100
                        print(f"进度: {results['total_files']}/{total_files} ({percent:.1f}%) - 已用时间: {elapsed:.1f}秒")

                    try:
                        # 分析挤出操作
                        extrude_type, has_extrude = self.analyze_extrude_operations(file_path)
                        if has_extrude:
                            results["extrude_types"][extrude_type] += 1
                            if extrude_type not in results["example_files"]:
                                results["example_files"][extrude_type] = file_path
                        else:
                            results["files_without_extrude"] += 1

                        # 统计face数量
                        face_count = self.count_faces(file_path)
                        if face_count > 5:
                            results["files_with_many_faces"] += 1
                            if len(results["many_faces_examples"]) < 5:
                                results["many_faces_examples"].append(file_path)

                        # 检查变换矩阵
                        transform_results = self.check_transforms(file_path)
                        for key, is_non_standard in transform_results.items():
                            if is_non_standard:
                                results["non_standard_transforms"][key] += 1
                                if key not in results["non_standard_examples"]:
                                    results["non_standard_examples"][key] = file_path

                    except Exception as e:
                        print(f"处理文件 {file_path} 时出错: {e}")

        return results

    def print_results(self, results: Dict):
        """打印分析结果"""
        print("\n===== 挤出操作类型统计 =====")
        print(f"总计扫描 OBJ 文件数: {results['total_files']}")
        print("\n挤出操作类型及数量:")
        for op_type, count in results["extrude_types"].most_common():
            print(f"  {op_type}: {count} 个文件")
            print(f"    示例文件: {results['example_files'][op_type]}")

        print(f"\n不包含挤出操作的文件数: {results['files_without_extrude']}")
        print(f"\nface数量超过5个的文件数: {results['files_with_many_faces']}")
        if results["many_faces_examples"]:
            print("\nface数量超过5个的示例文件:")
            for example in results["many_faces_examples"]:
                print(f"  {example}")

        print("\n===== 变换矩阵统计 =====")
        print("标准变换矩阵值:")
        for key, value in self.standard_transforms.items():
            print(f"  {key}: {value}")

        print("\n非标准变换矩阵数量:")
        for key, count in results["non_standard_transforms"].items():
            print(f"  {key} 不为标准值的文件数: {count}")
            if key in results["non_standard_examples"]:
                print(f"    示例文件: {results['non_standard_examples'][key]}")

def main():
    base_dir = '/home/malacca/3DLLM/data/cad_data_normalized'
    analyzer = OBJFileAnalyzer(base_dir)
    results = analyzer.analyze_all_files()
    analyzer.print_results(results)

if __name__ == "__main__":
    main() 