import os

def read_fe_coordinates(file_path):
    """
    Reads the coordinates of Fe atoms from a CONTCAR file.

    Parameters:
        file_path (str): Path to the CONTCAR file.

    Returns:
        list of tuple: A list of coordinates of Fe atoms.
    """
    fe_coordinates = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Read atomic species and count
        element_line = lines[5].split()
        count_line = lines[6].split()

        # Check if Fe is present
        if 'Fe' in element_line:
            fe_index = element_line.index('Fe')
            fe_count = int(count_line[fe_index])

            # Determine the starting line for coordinates
            start_line = 8  # Assuming "Selective Dynamics" is not present
            if 'Selective Dynamics' in lines[7]:
                start_line += 1

            # Find the Fe atoms
            total_atoms = sum(map(int, count_line))
            atom_start = start_line
            atom_end = atom_start + total_atoms

            current_count = 0
            for idx, element in enumerate(element_line):
                if idx < fe_index:
                    current_count += int(count_line[idx])

            fe_start = atom_start + current_count
            fe_end = fe_start + fe_count

            for line in lines[fe_start:fe_end]:
                coords = tuple(map(float, line.split()[:3]))
                fe_coordinates.append(coords)

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

    return fe_coordinates

def process_multiple_contcars(directory):
    """
    Processes all CONTCAR files in a directory to extract Fe coordinates.

    Parameters:
        directory (str): Path to the directory containing CONTCAR files.

    Returns:
        dict: A dictionary with filenames as keys and Fe coordinates as values.
    """
    results = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("contcar_"):
                file_path = os.path.join(root, file)
                fe_coordinates = read_fe_coordinates(file_path)
                results[file_path] = fe_coordinates

    return results

# Example usage:
directory_path = "python_geo/Fe_location/Fe/contcar_files"  # Replace with your directory containing CONTCAR files
fe_data = process_multiple_contcars(directory_path)
for file_path, coords in fe_data.items():
    print(f"File: {file_path}")
    for coord in coords:
        print(coord)