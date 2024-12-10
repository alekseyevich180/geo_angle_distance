import os
import glob

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

    coordinates_lines = lines[coordinate_start_line : coordinate_start_line + sum(atom_counts)]
    coordinates = [line.split()[:3] for line in coordinates_lines]

    atoms_coordinates = []
    current_index = 0
    for atom_type, count in zip(atom_types, atom_counts):
        for _ in range(count):
            atoms_coordinates.append((atom_type, list(map(float, coordinates[current_index]))))
            current_index += 1

    return atoms_coordinates

if __name__ == "__main__":
    folder_path = "."  # 当前目录
    file_pattern = os.path.join(folder_path, "CONTCAR_*.vasp")
    files = glob.glob(file_pattern)

    print(f"找到 {len(files)} 个文件: {files}")

    output_file = "results.txt"
    with open(output_file, "w", encoding="utf-8") as f:  # 设置文件写入编码为 UTF-8
        for file_path in files:
            f.write(f"\n处理文件: {file_path}\n")
            print(f"\n处理文件: {file_path}")
            try:
                atoms_coordinates = parse_poscar(file_path)

                f.write("所有原子和坐标:\n")
                print("所有原子和坐标:")
                for atom, coord in atoms_coordinates:
                    line = f"{atom}: {coord}\n"
                    f.write(line)
                    print(line.strip())

                if len(atoms_coordinates) > 1:
                    second_last_atom = atoms_coordinates[-2]
                    line = f"\n倒数第二个原子及其坐标:\n{second_last_atom[0]}: {second_last_atom[1]}\n"
                    f.write(line)
                    print(line.strip())
                else:
                    f.write("\n文件中原子不足两个，无法获取倒数第二个原子。\n")
                    print("\n文件中原子不足两个，无法获取倒数第二个原子。")
            except Exception as e:
                error_message = f"处理文件 {file_path} 时发生错误: {e}\n"
                f.write(error_message)
                print(error_message)

    print(f"\n所有结果已保存到文件 {output_file}")
