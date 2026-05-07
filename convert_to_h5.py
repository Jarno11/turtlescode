import os
import re
import subprocess
import pyvista as pv
import h5py
import numpy as np

# Configuration
tag = "{tag}"
mesh_path = "partitiondir/cylinder.pyfrm"
prefix = "{simname}"
input_dir = f"<trajectories_dir>/{{prefix}}"
output_dir = f"<dataset_dir>/{{tag}}/{{prefix}}"
subdivlevel = "4"

files = [f for f in os.listdir(input_dir) if f.endswith('.pyfrs') and prefix + "_ts" in f]


def get_timestamp(filename):
    """Extract the simulation timestamp from a .pyfrs filename."""
    match = re.search(r'ts([\d.]+)\.pyfrs', filename)
    return float(match.group(1)) if match else 0


files.sort(key=get_timestamp)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for index, filename in enumerate(files):
    input_file = os.path.join(input_dir, filename)
    output_file = os.path.join(output_dir, f"{{prefix}}_ts{{index}}.vtu")

    print(f"--- Processing {{filename}} ({{index + 1}}/{{len(files)}}) ---")

    # Export solution file from .pyfrs to high-order .vtu
    if os.path.exists(output_file):
        print(f"Skip Export (Exists): {{output_file}}")
    else:
        subprocess.run([
            "pyfr", "export", "-p", "single", "-k", subdivlevel,
            mesh_path, input_file, output_file
        ], check=True, capture_output=True)
        print(f"Exported: {{output_file}}")

    raw_grid = pv.read(output_file)

    # Remove duplicate solution nodes
    grid = raw_grid.clean(tolerance=1e-5)

    # Sort files into batches of 100 timesteps
    batch_start = (index // 100) * 100
    batch_h5 = os.path.join(output_dir, f"{{prefix}}_b{{batch_start}}_{{batch_start + 99}}.h5")

    coord_file = os.path.join(output_dir, f"{{prefix}}_coords.h5")

    if index == 0:
        # Save a template .vtu file and the mesh coordinates
        template_grid = grid.copy()
        template_grid.point_data['Density'] = np.ones(template_grid.n_points)
        template_grid.point_data['Pressure'] = np.ones(template_grid.n_points)
        template_grid.point_data['Velocity'] = np.ones((template_grid.n_points, 3))
        template_grid.save(os.path.join(output_dir, f"{{prefix}}_template.vtu"))

        with h5py.File(coord_file, 'w') as fc:
            for i, ax in enumerate(['x', 'y', 'z']):
                fc.create_dataset(ax, data=grid.points[:, i], compression="gzip")
        print(f"Saved coordinates file: {{coord_file}}")

    elif index == 20 or index % 50 == 0:
        # Periodically verify mesh consistency
        with h5py.File(coord_file, 'r') as fc:
            ref_points = np.column_stack((fc['x'][:], fc['y'][:], fc['z'][:]))
            if not np.allclose(grid.points, ref_points, atol=1e-8):
                raise ValueError(f"Mesh inconsistency detected at index {{index}}! "
                                 f"Current coordinates do not match the reference in {{coord_file}}.")
            else:
                print(f"Verification passed: mesh is consistent at index {{index}}")

    # Store solution fields in the batch HDF5 file
    with h5py.File(batch_h5, 'a') as f:
        group_name = f"sol_t{{index}}"
        if group_name in f:
            del f[group_name]
        grp = f.create_group(group_name)
        grp.create_dataset('rho', data=grid.point_data['Density'], compression="gzip")
        grp.create_dataset('p', data=grid.point_data['Pressure'], compression="gzip")
        vel = grid.point_data['Velocity']
        for i, v_comp in enumerate(['u', 'v', 'w']):
            grp.create_dataset(v_comp, data=vel[:, i], compression="gzip")

    print(f"Stored in batch: {{batch_h5}}")

    del raw_grid, grid
    if 'template_grid' in locals():
        del template_grid
    os.remove(output_file)
