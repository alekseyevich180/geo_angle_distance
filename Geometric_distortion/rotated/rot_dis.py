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
    file_pattern = os.path.join(folder_path, "IrO6.vasp")
    files = glob.glob(file_pattern)
    
    if not files:
        print("未找到匹配的文件 IrO6.vasp")
        exit()
    else:
        print(f"找到 {len(files)} 个文件: {files}")

    poscar_file = files[0]

    # 解析POSCAR文件
    atoms_coordinates, lines, atom_types, atom_counts, coordinate_start_line = parse_poscar(poscar_file)

    # 假设Ir为第一个原子类型，O为第二个原子类型（例如IrO6结构）
    ir_count = atom_counts[0]
    o_count = atom_counts[1] if len(atom_counts) > 1 else 0

    if ir_count != 1:
        raise ValueError("未能正确识别单一的中心Ir原子，请检查文件。")
    
    if o_count < 5:
        raise ValueError("氧原子数量不足（<5），无法按照示例逻辑定义平面和目标原子。")

    # 提取Ir和O坐标
    Ir = None
    O_list = []
    for atom, coord in atoms_coordinates:
        if atom.lower().startswith('ir') and Ir is None:
            Ir = np.array(coord)
        elif atom.lower().startswith('o'):
            O_list.append(np.array(coord))

    if Ir is None:
        raise ValueError("未找到Ir原子。")

    # 使用前4个O定义平面，并选取第5个O为目标旋转原子
    O1, O2, O3, O4 = O_list[0], O_list[1], O_list[2], O_list[3]
    O_target = O_list[4]

    # 定义平面法向量
    v1 = O2 - O1
    v2 = O3 - O1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)  # 归一化

    # O_target相对于Ir的向量和初始距离
    OM = O_target - Ir
    initial_distance = np.linalg.norm(OM)

    # 确定要修改的O原子行：假设atom_counts中Ir在前，O在后
    o_start_line = coordinate_start_line + ir_count
    # 第5个O对应行号：o_start_line + 4（下标从0开始）
    target_line_index = o_start_line + 4
    final_distance = 0.10623807321294945  # 目标缩短后的O-Ir距离

    # 对0到60度，每度一次旋转，并输出文件
    for angle in range(0, 61, 10):
        theta = math.radians(angle)
        R = rotation_matrix(normal, theta)
        OM_rotated = R.dot(OM)
        O_target_new = Ir + OM_rotated

        # 检查旋转后距离是否与初始距离相同
        #new_distance = np.linalg.norm(O_target_new - Ir)
        #if not math.isclose(initial_distance, new_distance, rel_tol=1e-10, abs_tol=1e-10):
            #print(f"警告：旋转前后距离发生改变 (初始: {initial_distance}, 新: {new_distance})")
        
        # 修改目标行坐标
        current_distance = np.linalg.norm(O_target_new - Ir)
        scale_factor = final_distance / current_distance
        O_target_final = Ir + OM_rotated * scale_factor

    # 修改目标行坐标至缩短后的坐标
        line_split = lines[target_line_index].split()
        line_split[0] = f"{O_target_final[0]:.16f}"
        line_split[1] = f"{O_target_final[1]:.16f}"
        line_split[2] = f"{O_target_final[2]:.16f}"
        new_line = " ".join(line_split) + "\n"
        original_line = lines[target_line_index]
        lines[target_line_index] = new_line

    # 输出文件名带角度标记
        output_filename = f"IrO6_modified_{angle}deg.vasp"
        with open(output_filename, "w") as f:
            f.writelines(lines)

    # 恢复原始行以便下个循环使用
        lines[target_line_index] = original_line

    print("已生成0-60度的旋转并缩短距离后的坐标文件。")

