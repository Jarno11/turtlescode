import gmsh
import subprocess
import os


def mshgen(execute, geo_path, msh_path):
    """Generate a 3D mesh from a .geo file using Gmsh."""
    if not execute:
        return

    print("Starting meshing: .geo -> .msh")
    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.open(geo_path)
        gmsh.model.mesh.generate(3)
        gmsh.write(msh_path)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        gmsh.finalize()
        print("Meshing completed.")


def pyfrmgen(execute, msh_path, pyfrm_path):
    """Import a .msh file into PyFR format (.pyfrm)."""
    if not execute:
        return

    print("Importing the mesh: .msh -> .pyfrm")
    subprocess.run(["pyfr", "import", msh_path, pyfrm_path], check=True)
    print("Import completed.")


def partitiongen(execute, pyfrm_path, partition_path):
    """Partition a .pyfrm mesh for parallel execution using SCOTCH."""
    if not execute:
        return

    work_dir = os.environ.get('WORK')
    os.environ['PYFR_SCOTCH_LIBRARY_PATH'] = f"{work_dir}/software/scotch-install/lib64/libscotch.so"
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = f"{work_dir}/software/scotch-install/lib64:{current_ld_path}"

    print("Partitioning the mesh...")
    subprocess.run([
        "pyfr", "partition", "-p", "scotch", "4",
        pyfrm_path, partition_path,
        "-e", "hex:3", "-e", "pri:2"
    ], check=True)
    print("Partitioning completed.")
