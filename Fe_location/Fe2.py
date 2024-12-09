from ase.io import read

def get_last_fe_position(poscar_file):
    # Read the POSCAR file using ASE
    structure = read(poscar_file, format='vasp')

    # Get the indices of all Fe atoms
    fe_indices = [i for i, atom in enumerate(structure) if atom.symbol == 'Fe']

    # Check if there are any Fe atoms
    if not fe_indices:
        raise ValueError("No Fe atoms found in the POSCAR file.")

    # Get the position of the last Fe atom
    last_fe_index = fe_indices[-1]
    last_fe_position = structure[last_fe_index].position

    return last_fe_position

# Example usage
if __name__ == "__main__":
    poscar_file = "POSCAR"  # Replace with your POSCAR file path
    try:
        position = get_last_fe_position(poscar_file)
        print(f"Position of the last Fe atom: {position}")
    except Exception as e:
        print(f"Error: {e}")
