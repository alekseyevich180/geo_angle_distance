import numpy as np
import os

def parse_poscar(poscar_path):
    """
    解析POSCAR/vasp文件，将每个原子类型与其坐标对应起来。
    返回格式:
    atoms_coordinates: [(atom_type, [x, y, z]), ...]
    lines(文件全部行), atom_types, atom_counts, coordinate_start_line
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

# -----------------------
# 主程序：使用 parse_poscar 从 IrO6.vasp 中读取数据，并进行随机结构生成
# -----------------------

# 输入文件
folder_path = "."
input_file = os.path.join(folder_path, "IrO6.vasp")
#input_file = "IrO6.vasp"
atoms_coordinates, lines, atom_types, atom_counts, coord_start = parse_poscar(input_file)

# 从POSCAR提取信息
scale = float(lines[1].strip())
lattice = []
for i in range(2, 5):
    vec = list(map(float, lines[i].split()))
    lattice.append(vec)
lattice = np.array(lattice) * scale

# 区分Ir和O原子并获取坐标(Direct坐标)
Ir_coords_direct = []
O_coords_direct = []

for atom_type, coord in atoms_coordinates:
    if atom_type == "Ir":
        Ir_coords_direct.append(coord)
    elif atom_type == "O":
        O_coords_direct.append(coord)

O_coords_direct = np.array(O_coords_direct)
Ir_coords_direct = np.array(Ir_coords_direct)  # 理论上只有一个Ir

# 将O原子direct坐标转为笛卡尔坐标
O_coords_cart = O_coords_direct @ lattice

# 计算O原子的质心
center = np.mean(O_coords_cart, axis=0)

# 计算O到质心的最大距离，用于定义球半径
distances = np.linalg.norm(O_coords_cart - center, axis=1)
R = np.max(distances)

num_structures = 60
output_dir = "./structures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 随机生成函数：在球体内随机产生点
def random_point_in_sphere(R):
    while True:
        x = np.random.uniform(-R, R)
        y = np.random.uniform(-R, R)
        z = np.random.uniform(-R, R)
        if x**2 + y**2 + z**2 <= R**2:
            return np.array([x, y, z])

for i in range(num_structures):
    # 在球体内选取一个随机点(设置0.8倍的R以适当缩小范围，可根据需要调整)
    metal_cart = center + random_point_in_sphere(R * 0.8)
    inv_lattice = np.linalg.inv(lattice)
    metal_direct = metal_cart @ inv_lattice

    # 输出文件(与POSCAR格式相同)
    lines_out = []
    lines_out.append(f"3D\\Atomistic - Random structure #{i+1}")
    lines_out.append(f"{1.0:.10f}")
    for vec in lattice:
        lines_out.append(" ".join(f"{x:15.10f}" for x in vec))
    lines_out.append(" ".join(atom_types))
    lines_out.append(" ".join(map(str, atom_counts)))
    lines_out.append("Direct")

    # 首先写金属原子(仅1个Ir)
    lines_out.append(" ".join(f"{x:15.9f}" for x in metal_direct))
    # 再写O原子
    for coord in O_coords_direct:
        lines_out.append(" ".join(f"{x:15.9f}" for x in coord))

    filename = os.path.join(output_dir, f"structure_{i+1:03d}.vasp")
    with open(filename, "w") as f:
        f.write("\n".join(lines_out) + "\n")

print(f"{num_structures} 个结构已生成至 {output_dir} 文件夹下。")
