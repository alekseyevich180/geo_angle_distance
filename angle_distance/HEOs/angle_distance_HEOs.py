import numpy as np
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

def convert_to_cartesian(coordinates, lattice_vectors):
    return np.dot(coordinates, lattice_vectors)

def read_poscar(file_path):
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
    
    print(f"Total atoms: {sum(element_counts)}")
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

def find_octachedral_center_and_neighbors(elements, atom_coordinates, center_elements, neighbor_element, min_distance, max_distance):
    center_and_neighbors = []
    
    for center_element in center_elements:
        centers = [atom_coordinates[i] for i, element in enumerate(elements) if element == center_element]
        neighbors = [atom_coordinates[i] for i, element in enumerate(elements) if element == neighbor_element]

        for center in centers:
            nearby_neighbors = [neighbor for neighbor in neighbors if min_distance < np.linalg.norm(center - neighbor) < max_distance]
            if len(nearby_neighbors) >= 3:  
                center_and_neighbors.append((center, nearby_neighbors))
    
    print(f"Total valid octahedral centers found: {len(center_and_neighbors)}")
    return center_and_neighbors

def main():
    file_path = 'angle_distance/POSCAR_cleaned'
    center_elements = ['Mn', 'Co', 'Ni', 'Fe']  # 高熵氧化物中的可能中心元素
    neighbor_element = 'O'  # 假设配位原子是 O
    min_distance = 1.6  # 配位原子选择的最小距离（单位Å）
    max_distance = 2.2  # 配位原子选择的最大距离（单位Å）

    elements, atom_coordinates, atom_names = read_poscar(file_path)
    centers_and_neighbors = find_octachedral_center_and_neighbors(elements, atom_coordinates, center_elements, neighbor_element, min_distance, max_distance)

    angle_deviations = []
    bond_length_distortions = []
    
    for center, neighbors in centers_and_neighbors:
        angle_deviation = calculate_angle_deviation(center, neighbors)
        bond_length_distortion = calculate_bond_length_distortion(center, neighbors)
        angle_deviations.append(angle_deviation)
        bond_length_distortions.append(bond_length_distortion)

    avg_angle_deviation = np.mean(angle_deviations) if angle_deviations else 0.0
    avg_bond_length_distortion = np.mean(bond_length_distortions) if bond_length_distortions else 0.0

    output_file_path = 'angle_distance/geo_high_entropy_txt'
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(f"高熵氧化物的平均键角偏离度: {avg_angle_deviation:.2f}°\n")
        f.write(f"高熵氧化物的平均键长畸变指数: {avg_bond_length_distortion:.4f}\n")

    print(f"高熵氧化物的平均键角偏离度: {avg_angle_deviation:.2f}°")
    print(f"高熵氧化物的平均键长畸变指数: {avg_bond_length_distortion:.4f}")
    print(f"结果已保存到 '{output_file_path}'")

if __name__ == "__main__":
    main()
