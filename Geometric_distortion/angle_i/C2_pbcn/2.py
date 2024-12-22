import os
import glob
import math
import numpy as np
import re

# Helper function to parse POSCAR files
def parse_poscar(poscar_path):
    """
    解析POSCAR文件，将每个原子类型与其坐标对应起来。
    """
    with open(poscar_path, 'r') as file:
        lines = file.readlines()

    atom_types = lines[5].split()
    atom_counts = list(map(int, lines[6].split()))
    coordinate_start_line = 8

    if lines[7].strip().lower() in ["selective dynamics", "s"]:
        coordinate_start_line += 1

    coordinates_lines = lines[coordinate_start_line: coordinate_start_line + sum(atom_counts)]
    coordinates = [line.split()[:3] for line in coordinates_lines]

    atoms_coordinates = []
    current_index = 0
    for atom_type, count in zip(atom_types, atom_counts):
        for _ in range(count):
            atoms_coordinates.append((atom_type, list(map(float, coordinates[current_index]))))
            current_index += 1

    return atoms_coordinates, lines, atom_types, atom_counts, coordinate_start_line

# Helper function to calculate plane normal
def calculate_plane_normal(p1, p2, p3):
    """
    计算由三个点定义的平面的法向量。
    """
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    return np.cross(v1, v2)

# Helper function to calculate angle
def calculate_angle(vec1, vec2):
    """
    计算两个向量之间的夹角（以度为单位）。
    """
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)

    if magnitude1 == 0 or magnitude2 == 0:
        raise ValueError("输入向量的模不能为零。")

    dot_product = np.dot(vec1, vec2)
    cos_theta = np.clip(dot_product / (magnitude1 * magnitude2), -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))

# Natural sort key for filenames
def natural_sort_key(s):
    """
    提取文件名中的数字部分，用于自然排序。
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# Main processing function
def process_files(folder_path):
    """
    处理文件夹中的所有.vasp文件，计算平面法向量与目标向量的夹角以及原子间夹角。
    """
    file_pattern = os.path.join(folder_path, "*.vasp")
    files = glob.glob(file_pattern)

    if not files:
        print("未找到.vasp文件，请检查文件夹。")
        return

    results = []

    for poscar_file in files:
        print(f"处理文件: {poscar_file}")
        atoms_coordinates, lines, atom_types, atom_counts, coordinate_start_line = parse_poscar(poscar_file)

        # 确认中心金属原子和氧原子
        metal_type = "Ir"  # 修改为POSCAR文件中金属原子的实际名称
        oxygen_type = "O"
        metal_indices = [i for i, atom in enumerate(atoms_coordinates) if atom[0] == metal_type]
        oxygen_indices = [i for i, atom in enumerate(atoms_coordinates) if atom[0] == oxygen_type]

        if not metal_indices:
            print(f"文件 {poscar_file} 中未找到金属原子 {metal_type}，跳过。")
            continue

        if len(oxygen_indices) < 6:
            print(f"文件 {poscar_file} 中氧原子数量不足，跳过。")
            continue

        # 选择特定的金属原子作为中心（例如第16号）
        central_metal_index = 8 - 1  # 替换为目标铱原子的索引
        if central_metal_index >= len(metal_indices):
            print(f"文件 {poscar_file} 中不存在指定的中心金属原子索引 {central_metal_index + 1}，跳过。")
            continue

        metal = np.array(atoms_coordinates[metal_indices[central_metal_index]][1])
        oxygens = [np.array(atoms_coordinates[i][1]) for i in oxygen_indices]

        # 计算平面法向量
        O16, O24, O23, O15 = oxygens[1], oxygens[2], oxygens[6], oxygens[5]  # 替换为实际索引
        normal = calculate_plane_normal(O16, O24, O23)

        # 计算目标向量与平面法向量夹角
        O6, O18 = oxygens[0], oxygens[4]  # 替换为实际索引
        v6 = O6 - metal
        v18 = O18 - metal

        angle_6 = calculate_angle(v6, normal)
        angle_18 = calculate_angle(v18, normal)

        results.append((poscar_file, angle_6, angle_18))

    # 按文件名自然排序
    results.sort(key=lambda x: natural_sort_key(x[0]))

    # 保存结果
    summary_file = os.path.join(folder_path, "summary_results.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        for poscar_file, angle_6, angle_18 in results:
            f.write(f"文件: {poscar_file}, O6夹角: {angle_6:.2f} 度, O18夹角: {angle_18:.2f} 度\n")

    print(f"所有文件处理完成，结果已保存到 {summary_file}")

if __name__ == "__main__":
    folder_path = os.getcwd()  # 当前工作目录

    if not os.path.exists(folder_path):
        print("文件夹路径不存在，请检查后重新输入。")
    else:
        process_files(folder_path)
