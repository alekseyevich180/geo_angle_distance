import numpy as np
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

def convert_to_cartesian(coordinates, lattice_vectors):
    """
    将分数坐标转换为实际坐标（笛卡尔坐标）。
    """
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
    atom_names = []  # 存储原子名称
    start_line = 8  # Assuming direct coordinates start at line 8

    for element, count in zip(elements, element_counts):
        for i in range(count):
            atom_coordinates.append(list(map(float, lines[start_line].strip().split())))
            atom_names.append(f"{element}{i + 1}")  # 为每个金属原子命名
            start_line += 1
            
    atom_coordinates = np.array(atom_coordinates)
    atom_coordinates = np.dot(atom_coordinates, lattice_vectors)
    
    # 生成每个原子的实际元素列表
    expanded_elements = []
    for element, count in zip(elements, element_counts):
        expanded_elements.extend([element] * count)
    
    #print(f"Elements: {elements}, Element counts: {element_counts}")
    print(f"Total atoms: {sum(element_counts)}")
    
    # 输出每个原子的坐标
    #for name, coord in zip(atom_names, atom_coordinates):
        #print(f"{name}: {coord}")

    return expanded_elements, element_counts, atom_coordinates, atom_names


def calculate_angle(v1, v2):
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_degrees = np.degrees(angle)
    #print(f"Calculating angle between vectors: {v1} and {v2}")
    #print(f"Cosine of angle: {cos_theta}, Angle (degrees): {angle_degrees}")
    return angle_degrees

def calculate_angle_deviation(center, neighbors):
    ideal_angles = [90, 180]
    deviations = []
    #print(f"Calculating angle deviations for center: {center} and neighbors: {neighbors}")
    
    for i in range(len(neighbors)):
        for j in range(i + 1, len(neighbors)):
            v1 = neighbors[i] - center
            v2 = neighbors[j] - center
            angle = calculate_angle(v1, v2)
            deviation = min([abs(angle - ideal) for ideal in ideal_angles])
            deviations.append(deviation)
            #print(f"Angle between neighbors {i} and {j}: {angle}, Deviation: {deviation}")
    
    avg_deviation = np.mean(deviations) if deviations else 0.0
    #print(f"Average angle deviation: {avg_deviation}")
    return avg_deviation

def calculate_bond_length_distortion(center, neighbors):
    bond_lengths = [np.linalg.norm(neighbor - center) for neighbor in neighbors]
    avg_bond_length = np.mean(bond_lengths)
    distortion_index = (
        np.mean([abs(bond - avg_bond_length) / avg_bond_length for bond in bond_lengths])
        if bond_lengths else 0.0
    )
    #print(f"Calculating bond length distortion for center: {center}")
    #print(f"Bond lengths: {bond_lengths}, Average bond length: {avg_bond_length}, Distortion index: {distortion_index}")
    return distortion_index

def find_octachedral_center_and_neighbors(elements, atom_coordinates, center_element, neighbor_element, min_distance, max_distance):
    
    #print(f"Looking for center element '{center_element}' and neighbor element '{neighbor_element}'")
    #print(f"Elements in POSCAR: {elements}")
    centers = [atom_coordinates[i] for i, element in enumerate(elements) if element == center_element]
    neighbors = [atom_coordinates[i] for i, element in enumerate(elements) if element == neighbor_element]
    
    center_and_neighbors = []
    #print(f"Finding octahedral centers and neighbors with elements: {center_element} and {neighbor_element}")
    #print(f"Found centers: {centers}, Found neighbors: {neighbors}")
    
    for center in centers:
        nearby_neighbors = [neighbor for neighbor in neighbors if min_distance < np.linalg.norm(center - neighbor) < max_distance]
        #print(f"Center: {center}, Nearby neighbors (within range): {nearby_neighbors}")
        
        # 放宽条件，允许至少 3 个邻居
        if len(nearby_neighbors) >= 3:  
            center_and_neighbors.append((center, nearby_neighbors))
            #print(f"Valid octahedral center found: {center} with neighbors: {nearby_neighbors}")
    
    print(f"Total valid octahedral centers found: {len(center_and_neighbors)}")
    print(f"找到的中心原子数量: {len(centers)}")
    print(f"找到的邻居原子数量: {len(neighbors)}")
    #print(f"找到的中心原子和邻居: {center_and_neighbors}")
    return center_and_neighbors

def main():
    file_path = 'angle_distance/POSCAR_cleaned'
    center_element = 'Ir'  # 假设中心原子是 Sn
    neighbor_element = 'O'  # 假设配位原子是 O
    min_distance = 1.6  # 配位原子选择的最小距离（单位Å）
    max_distance = 2  # 配位原子选择的最大距离（单位Å）

    elements, element_counts, atom_coordinates, atom_names = read_poscar(file_path)
    centers_and_neighbors = find_octachedral_center_and_neighbors(elements, atom_coordinates, center_element, neighbor_element, min_distance, max_distance)

    angle_deviations = []
    bond_length_distortions = []
    
    for center, neighbors in centers_and_neighbors:
        angle_deviation = calculate_angle_deviation(center, neighbors)
        bond_length_distortion = calculate_bond_length_distortion(center, neighbors)
        angle_deviations.append(angle_deviation)
        bond_length_distortions.append(bond_length_distortion)

    avg_angle_deviation = np.mean(angle_deviations) if angle_deviations else 0.0
    avg_bond_length_distortion = np.mean(bond_length_distortions) if bond_length_distortions else 0.0



    output_file_path = 'angle_distance/geo_Ir_txt'  # 输出文件路径
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(f"八面体结构的平均键角偏离度: {avg_angle_deviation:.2f}°\n")
        f.write(f"八面体结构的平均键长畸变指数: {avg_bond_length_distortion:.4f}\n")

    print(f"八面体结构的平均键角偏离度: {avg_angle_deviation:.2f}°\n")
    print(f"八面体结构的平均键长畸变指数: {avg_bond_length_distortion:.4f}\n")
    print(f"结果已保存到 '{output_file_path}'")

if __name__ == "__main__":
    main()
