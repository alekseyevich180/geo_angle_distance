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

# Helper function to calculate rotation matrix
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

# Helper function to calculate plane normal
def calculate_plane_normal(p1, p2, p3):
    """
    计算由三个点定义的平面的法向量。
    """
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    return np.cross(v1, v2)

# Helper function to calculate angle
def calculate_angle(metal, oxygen1, oxygen2):
    """
    计算金属原子为顶点，两个氧原子为端点的夹角。
    """
    vec1 = np.array(oxygen1) - np.array(metal)
    vec2 = np.array(oxygen2) - np.array(metal)

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

        if O_count < 6:
            print(f"文件 {poscar_file} 的氧原子数量不足，跳过。")
            continue

        metal = None
        oxygens = []
        for atom, coord in atoms_coordinates:
            if atom == M_type and metal is None:
                metal = np.array(coord)
            elif atom == O_type:
                oxygens.append(np.array(coord))

        if metal is None or len(oxygens) < 6:
            print(f"文件 {poscar_file} 的原子坐标不足，跳过。")
            continue

        # 计算平面法向量与目标向量的夹角
        O1, O2, O3, O4 = oxygens[0], oxygens[1], oxygens[2], oxygens[3]
        O_target = oxygens[4]
        normal = calculate_plane_normal(O1, O2, O3)
        OM = O_target - metal

        magnitude1 = np.linalg.norm(normal)
        magnitude2 = np.linalg.norm(OM)

        if magnitude1 == 0 or magnitude2 == 0:
            raise ValueError("输入向量的模不能为零。")

        dot_product = np.dot(normal, OM)
        cos_theta = np.clip(dot_product / (magnitude1 * magnitude2), -1.0, 1.0)
        vector_angle = math.degrees(math.acos(cos_theta))
        surface_angle =  vector_angle -90

        # 计算金属原子与其他氧原子的夹角
        O5, O6 = oxygens[4], oxygens[5]
        angle = calculate_angle(metal, O5, O6)

        results.append((poscar_file, surface_angle, angle))

    # 按文件名自然排序
    results.sort(key=lambda x: natural_sort_key(x[0]))

    # 保存结果
    summary_file = os.path.join(folder_path, "summary_results.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        for poscar_file, surface_angle, angle in results:
            f.write(f"文件: {poscar_file}, 平面夹角:, {surface_angle:.2f} , 原子夹角: , {angle:.2f} \n")

    print(f"所有文件处理完成，结果已保存到 {summary_file}")

if __name__ == "__main__":
    folder_path = os.getcwd()

    if not os.path.exists(folder_path):
        print("文件夹路径不存在，请检查后重新输入。")
    else:
        process_files(folder_path)
