import os
import glob
import math
import numpy as np
import re
from atom_location import parse_poscar
from utils import calculate_plane_normal


def rotation_matrix(axis, theta):
    """
    给定旋转轴(axis)和旋转角度theta(弧度)，返回绕该轴旋转theta的旋转矩阵(Rodrigues公式)。
    axis应为归一化后的向量。
    """
    axis = axis / np.linalg.norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    rot = np.array([
        [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]
    ])
    return rot


def natural_sort_key(s):
    """
    提取文件名中的数字部分，用于自然排序。
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def process_all_files(folder_path):
    """
    处理文件夹中的所有.vasp文件，计算已存在旋转后的向量与定义平面的夹角。
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

        if len(atom_types) < 2 or len(atoms_coordinates) < 6:
            print(f"文件 {poscar_file} 的原子信息不足，跳过。")
            continue

        M_type = atom_types[0]
        O_type = atom_types[1]
        M_count = atom_counts[0]
        O_count = atom_counts[1] if len(atom_counts) > 1 else 0

        if M_count != 1:
            print(f"文件 {poscar_file} 中未找到单一的中心金属原子，跳过。")
            continue

        if O_count < 5:
            print(f"文件 {poscar_file} 的氧原子数量不足，跳过。")
            continue

        # 获取金属原子和氧原子的坐标
        M = None
        O_list = []
        for atom, coord in atoms_coordinates:
            if atom == M_type and M is None:
                M = np.array(coord)
            elif atom == O_type:
                O_list.append(np.array(coord))

        if M is None or len(O_list) < 5:
            print(f"文件 {poscar_file} 的原子坐标不足，跳过。")
            continue

        # 使用前4个O原子定义平面，选取第5个O原子为目标
        O1, O2, O3, O4 = O_list[0], O_list[1], O_list[2], O_list[3]
        O_target = O_list[4]

        # 定义平面法向量
        normal = calculate_plane_normal(O1, O2, O3)

        # O_target相对于M的向量
        OM = O_target - M

        magnitude1 = np.linalg.norm(normal)
        magnitude2 = np.linalg.norm(OM)

        if magnitude1 == 0 or magnitude2 == 0:
            raise ValueError("输入向量的模不能为零。")

        dot_product = np.dot(normal, OM)
        cos_theta = np.clip(dot_product / (magnitude1 * magnitude2), -1.0, 1.0)
        vector_angle = math.degrees(math.acos(cos_theta))
        surface_angle = 90 - vector_angle

        results.append((poscar_file, surface_angle))

    # 按文件名自然排序
    results.sort(key=lambda x: natural_sort_key(x[0]))

    # 输出结果到summary文件
    summary_file = os.path.join(folder_path, "summary_results.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        for poscar_file, angle in results:
            f.write(f"文件: {poscar_file}, 向量夹角: {angle:.2f} 度\n")

    print("所有文件处理完成，结果已保存。")


if __name__ == "__main__":
    folder_path = os.getcwd()

    if not os.path.exists(folder_path):
        print("文件夹路径不存在，请检查后重新输入。")
    else:
        process_all_files(folder_path)
