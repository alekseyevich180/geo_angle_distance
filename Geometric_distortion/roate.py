import os
import glob
import math
import numpy as np

def parse_poscar(poscar_path):
    """
    解析POSCAR/vasp文件，将每个原子类型与其坐标对应起来。
    返回格式: [(atom_type, [x, y, z]), ...]
    """
    with open(poscar_path, 'r') as file:
        lines = file.readlines()

    atom_types = lines[5].split()
    atom_counts = list(map(int, lines[6].split()))
    coordinate_start_line = 8

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

    return atoms_coordinates

def rotation_matrix(axis, theta):
    """
    给定旋转轴(axis)和旋转角度theta，返回绕该轴旋转theta的旋转矩阵(Rodrigues公式)。
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
    else:
        print(f"找到 {len(files)} 个文件: {files}")

    # 假设我们使用的文件为IrO6.vasp，如果有多个则只处理第一个
    if files:
        poscar_file = files[0]

        atoms_coordinates = parse_poscar(poscar_file)

        # 找出Ir原子和O原子索引与坐标（这里假设一个Ir在前面）
        # 实际可根据结构特定判断，这里只是简单假设第一个类型为Ir，后面为O
        ir_coords = [coord for atom, coord in atoms_coordinates if atom.lower().startswith('ir')]
        o_coords = [coord for atom, coord in atoms_coordinates if atom.lower().startswith('o')]

        if len(ir_coords) != 1:
            raise ValueError("未能正确识别单一的中心Ir原子。请检查POSCAR文件。")
        
        if len(o_coords) < 5:
            raise ValueError("氧原子不足以定义平面和进行旋转，请检查数据。")
        
        Ir = np.array(ir_coords[0])
        # 选择前4个O原子定义平面
        O1 = np.array(o_coords[0])
        O2 = np.array(o_coords[1])
        O3 = np.array(o_coords[2])
        O4 = np.array(o_coords[3])

        # 选第5个O为要旋转的目标原子
        O_target = np.array(o_coords[4])

        # 定义平面法向量
        v1 = O2 - O1
        v2 = O3 - O1
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)  # 归一化

        # 计算O_target相对于Ir的矢量
        OM = O_target - Ir
        distance = np.linalg.norm(OM)

        # 选择一个旋转角度theta（弧度制），例如旋转10度
        theta = np.deg2rad(2)  # 将10度转换为弧度

        # 计算旋转矩阵
        R = rotation_matrix(normal, theta)

        # 对OM进行旋转
        OM_rotated = R.dot(OM)

        # 新的目标O坐标
        O_target_new = Ir + OM_rotated

        # 检查旋转后距离是否保持不变
        new_distance = np.linalg.norm(O_target_new - Ir)
        print(f"旋转前距离: {distance:.6f}, 旋转后距离: {new_distance:.6f}")

        # 将结果写回新文件中(可选)
        # 假设POSCAR格式相同，只修改第五个O原子的坐标
        # 这里仅示意写文件过程，实际需考虑保持文件头部信息不变
        with open("IrO6_modified.vasp", "w") as f:
            # 简单示意：复制原文件内容，然后替换第5个O坐标
            # 实际应完善：复制头部、晶格参数、原子序列行，然后替换坐标行
            with open(poscar_file, 'r') as orig:
                lines = orig.readlines()
            
            # 根据之前parse_poscar的逻辑，第8行起为坐标
            # 假设无Selective Dynamics，这里不通用，只作示意
            # 如果有Selective Dynamics需调整行号
            coordinate_start_line = 8
            if lines[7].strip().lower() in ["selective dynamics", "s"]:
                coordinate_start_line += 1
            
            total_atoms = len(atoms_coordinates)
            # 修改第5个O原子的行：假设Ir在前面，O后面排列，具体根据POSCAR顺序而定
            # 假设atom_counts中Ir为1，O为6，那么:
            # 行索引：Ir(1行)是coordinate_start_line,
            # O1行：coordinate_start_line+1
            # O2行：coordinate_start_line+2
            # ...
            # O5行：coordinate_start_line+5  (待修改的那一行)

            # 找出POSCAR的atom_counts
            atom_types = lines[5].split()
            atom_counts = list(map(int, lines[6].split()))
            
            # 确定O原子在坐标中的起始行索引
            ir_count = atom_counts[0]  # 假设Ir在atom_types[0]
            # O在atom_types[1], count = atom_counts[1]
            # O坐标行起始：coordinate_start_line + ir_count
            o_start_line = coordinate_start_line + ir_count
            # 第5个O对应行号：o_start_line + 4（因从0计起）
            target_line_index = o_start_line + 4

            # 修改行内容
            new_line = f"{O_target_new[0]:.16f} {O_target_new[1]:.16f} {O_target_new[2]:.16f}\n"
            lines[target_line_index] = new_line

            # 将修改后的POSCAR写入新文件
            f.writelines(lines)

        print("已将修改后的结果写入 IrO6_modified.vasp 文件。")
