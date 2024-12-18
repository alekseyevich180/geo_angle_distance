import os
import glob
import math

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

    return atoms_coordinates, lines, atom_types, atom_counts, coordinate_start_line

def calculate_distance(atom1, atom2):
    """
    计算两个原子之间的距离。

    参数:
        - atom1: tuple，包含原子类型和坐标 (atom_type, [x, y, z])。
        - atom2: tuple，包含原子类型和坐标 (atom_type, [x, y, z])。

    返回:
        - distance: float，两个原子之间的距离。
    """
    coord1 = atom1[1]
    coord2 = atom2[1]
    distance = math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(coord1, coord2)))
    
    return distance    
