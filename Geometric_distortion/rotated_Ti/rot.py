import os
import glob
import math
import numpy as np

def parse_poscar(poscar_path):
    """
    解析POSCAR/vasp文件，将每个原子类型与其坐标对应起来。
    返回格式: [(atom_type, [x, y, z]), ...], lines(文件全部行), atom_types, atom_counts, coordinate_start_line
    """
    with open(poscar_path, 'r') as file:
        lines = file.readlines()

    atom_types = lines[5].split()
    atom_counts = list(map(int, lines[6].split()))
    coordinate_start_line = 8

    # 判断是否存在Selective Dynamics行
    if lines[7].strip().lower() in ["selective dynamics", "s"]:
        coordinate_start_line += 1

    total_atoms = sum(atom_counts)
    coordinates_lines = lines[coordinate_start_line : coordinate_start_line + total_atoms]
    coordinates = [line.split()[:3] for line in coordinates_lines]

    atoms_coordinates = []
    current_index = 0
    for atom_type, count in zip(atom_types, atom_counts):
        for _ in range(count):
            atoms_coordinates.append((atom_type, list(map(float, coordinates[current_index]))))
            current_index += 1

    return atoms_coordinates, lines, atom_types, atom_counts, coordinate_start_line

def rotation_matrix(axis, theta):
    """
    给定旋转轴(axis)和旋转角度theta(弧度)，返回绕该轴旋转theta的旋转矩阵(Rodrigues公式)。
    axis应为归一化后的向量。
    """
    axis = axis / np.linalg.norm(axis)
    a = math.cos(theta/2.0)
    b, c, d = -axis * math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    rot = np.array([[aa+bb-cc-dd, 2*(bc+ad),   2*(bd-ac)],
                    [2*(bc-ad),   aa+cc-bb-dd, 2*(cd-ab)],
                    [2*(bd+ac),   2*(cd+ab),   aa+dd-bb-cc]])
    return rot

if __name__ == "__main__":
    folder_path = "."
    # 修改为您实际的POSCAR文件名，如 "RuO6.vasp" 或 "SnO6.vasp"。此处以RuO6.vasp为例：
    file_pattern = os.path.join(folder_path, "TiO6.vasp")
    files = glob.glob(file_pattern)
    
    if not files:
        print("未找到匹配的文件 TiO6.vasp，请修改文件名以匹配您需要处理的对象。")
        exit()
    else:
        print(f"找到 {len(files)} 个文件: {files}")

    poscar_file = files[0]

    # 解析POSCAR文件
    atoms_coordinates, lines, atom_types, atom_counts, coordinate_start_line = parse_poscar(poscar_file)

    # 假设MO6结构：第一个原子类型为M(如Ru、Sn、Ti、Ir等)，第二个原子类型为O
    M_type = atom_types[0]  # 中心金属原子类型
    O_type = atom_types[1]  # 氧原子类型
    M_count = atom_counts[0]
    O_count = atom_counts[1] if len(atom_counts) > 1 else 0

    if M_count != 1:
        raise ValueError("未能正确识别单一的中心金属原子，请检查文件。")
    
    if O_count < 5:
        raise ValueError("氧原子数量不足（<5），无法定义平面和目标原子。")

    # 提取M和O坐标
    M = None
    O_list = []
    for atom, coord in atoms_coordinates:
        if atom == M_type and M is None:
            M = np.array(coord)
        elif atom == O_type:
            O_list.append(np.array(coord))

    if M is None:
        raise ValueError(f"未找到{M_type}原子。")
    
    distance = np.linalg.norm(O_list[0] - M)
    print(f"1号氧原子与中心{M_type}原子的距离为: {distance}")

    # 使用前4个O定义平面，并选取第5个O为目标旋转原子
    O1, O2, O3, O4 = O_list[0], O_list[1], O_list[2], O_list[3]
    O_target = O_list[4]

    # 定义平面法向量
    v1 = O2 - O1
    v2 = O3 - O1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)  # 归一化

    # O_target相对于M的向量和初始距离
    OM = O_target - M
    initial_distance = np.linalg.norm(OM)

    # 确定要修改的O原子行：假设M在前，O在后
    o_start_line = coordinate_start_line + M_count
    # 第5个O对应行号：o_start_line + 4（下标从0开始）
    target_line_index = o_start_line + 4

    # 请根据实际情况修改final_distance为所需的目标距离
    final_distance = 0.1009226579281015

    # 对0到60度，每隔20度一次旋转，并输出文件
    for angle in range(0, 61, 1):
        theta = math.radians(angle)
        R = rotation_matrix(normal, theta)
        OM_rotated = R.dot(OM)
        O_target_new = M + OM_rotated

        # 缩放使旋转后O-M距离为目标距离
        current_distance = np.linalg.norm(O_target_new - M)
        scale_factor = final_distance / current_distance
        O_target_final = M + OM_rotated * scale_factor

        # 修改目标行坐标至缩短后的坐标
        line_split = lines[target_line_index].split()
        line_split[0] = f"{O_target_final[0]:.16f}"
        line_split[1] = f"{O_target_final[1]:.16f}"
        line_split[2] = f"{O_target_final[2]:.16f}"
        new_line = " ".join(line_split) + "\n"
        original_line = lines[target_line_index]
        lines[target_line_index] = new_line

        # 输出文件名带角度标记（使用M_type来生成文件名）
        output_filename = f"{M_type}O6_{angle}.vasp"
        with open(output_filename, "w") as f:
            f.writelines(lines)

        # 恢复原始行以便下个循环使用
        lines[target_line_index] = original_line

    print("已生成0-60度的旋转并缩短距离后的MO6坐标文件。")
