import numpy as np
import os
from scipy.spatial import ConvexHull

def parse_poscar(poscar_path):
    """
    解析POSCAR/vasp文件，将每个原子类型与其坐标对应起来。
    返回: atoms_coordinates, lines, atom_types, atom_counts, coordinate_start_line
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

folder_path = "."
input_file = os.path.join(folder_path, "IrO6.vasp")
atoms_coordinates, lines, atom_types, atom_counts, coord_start = parse_poscar(input_file)

# 读取晶格
scale = float(lines[1].strip())
lattice = []
for i in range(2, 5):
    vec = list(map(float, lines[i].split()))
    lattice.append(vec)
lattice = np.array(lattice) * scale

# 区分Ir和O原子
Ir_coords_direct = []
O_coords_direct = []

for atom_type, coord in atoms_coordinates:
    if atom_type == "Ir":
        Ir_coords_direct.append(coord)
    elif atom_type == "O":
        O_coords_direct.append(coord)

O_coords_direct = np.array(O_coords_direct)
Ir_coords_direct = np.array(Ir_coords_direct)  # 通常只有1个Ir

# 将O原子坐标转换为笛卡尔坐标
O_coords_cart = O_coords_direct @ lattice

# 计算O原子的质心(中心点)
center = np.mean(O_coords_cart, axis=0)

# 使用ConvexHull找到八面体面
hull = ConvexHull(O_coords_cart)

# 计算内切球半径：对每个面计算点到面的距离，取最小值
def point_plane_distance(point, face_points):
    # face_points为构成面的三个顶点坐标
    p1, p2, p3 = face_points
    # 面的法向量
    normal = np.cross(p2 - p1, p3 - p1)
    normal /= np.linalg.norm(normal)
    # 距离为 |(point - p1)·normal|
    dist = abs(np.dot((point - p1), normal))
    return dist

# 找出所有面的距离
distances_to_faces = []
for simplex in hull.simplices:
    face_points = O_coords_cart[simplex]
    dist = point_plane_distance(center, face_points)
    distances_to_faces.append(dist)

# 内切球半径为距离的最小值
R_in_sphere = min(distances_to_faces)

# 我们将金属原子在内切球中随机分布
num_structures = 600
output_dir = "./structures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def random_point_in_sphere(R):
    # 在半径R的球内均匀生成点（拒绝采样法）
    while True:
        x = np.random.uniform(-R, R)
        y = np.random.uniform(-R, R)
        z = np.random.uniform(-R, R)
        if x**2 + y**2 + z**2 <= R**2:
            return np.array([x, y, z])

for i in range(num_structures):
    # 在内切球内生成随机点
    metal_cart = center + random_point_in_sphere(R_in_sphere)
    inv_lattice = np.linalg.inv(lattice)
    metal_direct = metal_cart @ inv_lattice

    # 输出文件
    lines_out = []
    lines_out.append(f"3D\\Atomistic - Random structure #{i+1}")
    lines_out.append(f"{1.0:.10f}")
    for vec in lattice:
        lines_out.append(" ".join(f"{x:15.10f}" for x in vec))
    lines_out.append(" ".join(atom_types))
    lines_out.append(" ".join(map(str, atom_counts)))
    lines_out.append("Direct")

    # 金属原子(仅1个Ir)
    lines_out.append(" ".join(f"{x:15.9f}" for x in metal_direct))
    # O原子
    for coord in O_coords_direct:
        lines_out.append(" ".join(f"{x:15.9f}" for x in coord))

    filename = os.path.join(output_dir, f"structure_{i+1:03d}.vasp")
    with open(filename, "w") as f:
        f.write("\n".join(lines_out) + "\n")

print(f"{num_structures} 个结构已生成至 {output_dir} 文件夹下，其中金属原子位于八面体的内切球内。")
