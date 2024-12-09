def parse_poscar(poscar_path):
    """
    解析POSCAR文件，将每个原子类型与其坐标对应起来。

    参数:
    - poscar_path: POSCAR文件路径

    返回:
    - atoms_coordinates: 一个列表，每个元素是 (原子类型, 坐标) 的元组
    """
    with open(poscar_path, 'r') as file:
        lines = file.readlines()

    # 读取原子类型和数量
    atom_types = lines[5].split()
    atom_counts = list(map(int, lines[6].split()))

    # 获取起始行号（坐标开始于第8行）
    coordinate_start_line = 8

    # 检查是否有选择性动力学（Selective dynamics）行
    if lines[7].strip().lower() in ["selective dynamics", "s"]:
        coordinate_start_line += 1

    # 获取原子坐标
    coordinates_lines = lines[coordinate_start_line : coordinate_start_line + sum(atom_counts)]
    coordinates = [line.split()[:3] for line in coordinates_lines]  # 提取前三列为坐标

    # 将原子类型与坐标一一对应
    atoms_coordinates = []
    current_index = 0
    for atom_type, count in zip(atom_types, atom_counts):
        for _ in range(count):
            atoms_coordinates.append((atom_type, list(map(float, coordinates[current_index]))))
            current_index += 1

    return atoms_coordinates


# 示例用法
if __name__ == "__main__":
    poscar_path = "POSCAR"  # 替换为实际POSCAR文件路径
    atoms_coordinates = parse_poscar(poscar_path)

    # 输出每个原子和对应的坐标
    print("所有原子和坐标:")
    for atom, coord in atoms_coordinates:
        print(f"{atom}: {coord}")

    # 输出倒数第二个原子
    if len(atoms_coordinates) > 1:
        second_last_atom = atoms_coordinates[-2]
        print("\n倒数第二个原子及其坐标:")
        print(f"{second_last_atom[0]}: {second_last_atom[1]}")
    else:
        print("\nPOSCAR中原子不足两个，无法获取倒数第二个原子。")
