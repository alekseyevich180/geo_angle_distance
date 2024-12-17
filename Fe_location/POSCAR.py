import numpy as np

def read_vasp(file_path):
    """
    Reads a POSCAR or .vasp file and extracts lattice vectors, atomic types, and coordinates.
    Converts fractional coordinates to Cartesian coordinates.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Scale factor
    scale = float(lines[1].strip())

    # Lattice vectors
    lattice_vectors = []
    for i in range(2, 5):
        vector = np.array([float(x) for x in lines[i].split()]) * scale
        lattice_vectors.append(vector)
    lattice_vectors = np.array(lattice_vectors)

    # Atomic types and number of atoms
    atom_types = lines[5].split()
    atom_numbers = [int(x) for x in lines[6].split()]

    # Total number of atoms
    total_atoms = sum(atom_numbers)

    # Coordinate system (Direct/Fractional or Cartesian)
    coord_type = lines[7].strip().lower()
    if 'direct' in coord_type or 'fractional' in coord_type:
        fractional = True
    elif 'cartesian' in coord_type:
        fractional = False
    else:
        raise ValueError("Coordinate type not recognized (must be 'Direct' or 'Cartesian').")

    # Atomic positions
    positions = []
    for i in range(8, 8 + total_atoms):
        pos = [float(x) for x in lines[i].split()[:3]]
        positions.append(pos)
    positions = np.array(positions)

    # Convert to Cartesian coordinates if necessary
    if fractional:
        cartesian_positions = np.dot(positions, lattice_vectors)
    else:
        cartesian_positions = positions

    # Output results
    print("Lattice Vectors:")
    print(lattice_vectors)
    print("\nAtomic Types:", atom_types)
    print("Number of Atoms:", atom_numbers)
    print("\nCartesian Coordinates (converted):")
    for i, pos in enumerate(cartesian_positions):
        print(f"Atom {i+1}: {pos}")

    return lattice_vectors, atom_types, atom_numbers, cartesian_positions

# Example usage
file_path = "POSCAR"  # Change to your POSCAR or .vasp file path
read_vasp(file_path)
