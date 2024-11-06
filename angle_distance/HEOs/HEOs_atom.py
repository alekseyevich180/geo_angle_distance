import numpy as np
import os
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

def convert_to_cartesian(coordinates, lattice_vectors):
    return np.dot(coordinates, lattice_vectors)

def read_contcar(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    scale_factor = float(lines[1].strip())
    lattice_vectors = np.array([list(map(float, lines[i].strip().split())) for i in range(2, 5)])
    lattice_vectors *= scale_factor
    
    elements = lines[5].strip().split()
    element_counts = list(map(int, lines[6].strip().split()))
    
    atom_coordinates = []
    atom_names = []
    start_line = 8

    for element, count in zip(elements, element_counts):
        for i in range(count):
            atom_coordinates.append(list(map(float, lines[start_line].strip().split())))
            atom_names.append(f"{element}{i + 1}")
            start_line += 1
            
    atom_coordinates = np.array(atom_coordinates)
    atom_coordinates = np.dot(atom_coordinates, lattice_vectors)
    
    expanded_elements = []
    for element, count in zip(elements, element_counts):
        expanded_elements.extend([element] * count)
    
    return expanded_elements, atom_coordinates, atom_names

def calculate_angle(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return np.degrees(angle)

def calculate_angle_deviation(center, neighbors):
    ideal_angles = [90, 180]
    deviations = []
    
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            v1 = neighbors[i] - center
            v2 = neighbors[j] - center
            angle = calculate_angle(v1, v2)
            deviation = min([abs(angle - ideal) for ideal in ideal_angles])
            deviations.append(deviation)
    
    return np.mean(deviations) if deviations else 0.0

def calculate_bond_length_distortion(center, neighbors):
    bond_lengths = [np.linalg.norm(neighbor - center) for neighbor in neighbors]
    avg_bond_length = np.mean(bond_lengths)
    distortion_index = (
        np.mean([abs(bond - avg_bond_length) / avg_bond_length for bond in bond_lengths])
        if bond_lengths else 0.0
    )
    return distortion_index

def find_specific_center_and_neighbors(elements, atom_coordinates, atom_names, specific_center_element, neighbor_element, min_distance, max_distance):
    specific_center_index = None
    
    for i, atom_name in enumerate(atom_names):
        if atom_name == specific_center_element:
            specific_center_index = i
            break
    
    if specific_center_index is None:
        print(f"指定的中心原子 '{specific_center_element}' 未找到。")
        return None, None
    
    center = atom_coordinates[specific_center_index]
    neighbors = [atom_coordinates[i] for i, element in enumerate(elements) if element == neighbor_element and min_distance < np.linalg.norm(center - atom_coordinates[i]) < max_distance]
    
    return center, neighbors

def process_specific_atom(directory_path, specific_center_element):
    neighbor_element = 'O'  # 假设配位原子是 O
    min_distance = 1.6  # 配位原子选择的最小距离（单位Å）
    max_distance = 2.2  # 配位原子选择的最大距离（单位Å）
    
    output_file_path = os.path.join(directory_path, f"{specific_center_element}_geo_results.txt")
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for file_name in sorted(os.listdir(directory_path)):
            if re.match(r"CONTCAR_\d+$", file_name):  # Match files with names like CONTCAR_2, etc.
                file_path = os.path.join(directory_path, file_name)
                elements, atom_coordinates, atom_names = read_contcar(file_path)
                
                center, neighbors = find_specific_center_and_neighbors(
                    elements, atom_coordinates, atom_names, specific_center_element, neighbor_element, min_distance, max_distance
                )

                if center is not None and neighbors:
                    angle_deviation = calculate_angle_deviation(center, neighbors)
                    bond_length_distortion = calculate_bond_length_distortion(center, neighbors)
                    
                    f.write(f"{file_name} - {specific_center_element} 八面体平均键角偏离度: {angle_deviation:.2f}°\n")
                    f.write(f"{file_name} - {specific_center_element} 八面体平均键长畸变指数: {bond_length_distortion:.4f}\n\n")
                    
                    print(f"{file_name} - {specific_center_element} 八面体平均键角偏离度: {angle_deviation:.2f}°")
                    print(f"{file_name} - {specific_center_element} 八面体平均键长畸变指数: {bond_length_distortion:.4f}")
                else:
                    print(f"{file_name} - 未找到有效的配位原子或指定的中心原子。")

        print(f"所有结果已保存到 '{output_file_path}'")

if __name__ == "__main__":
    directory_path = 'angle_distance/HEOs/CONTCARs/CONTCARs'
    specific_center_element = 'Co1'  # Specify the atom, e.g., 'Mn1' for only this atom's polyhedron calculations
    process_specific_atom(directory_path, specific_center_element)
