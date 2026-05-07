from pathlib import Path
import numpy as np
import geo
import meshing_utils
import shutil
import os


def setup_simulation(directory_loc, params):
    inflow_deg = int(np.round(params["inflow_angle"] * 180 / np.pi))
    n_str = "inf" if np.isinf(params["n"]) else str(params["n"])
    a_str = str(params["a"]).replace(".", "p")
    jobname = f"Re{params['Re']}_alpha{inflow_deg}_a{a_str}_n{n_str}"
    params.update({"simname": jobname})
    directory_string = os.path.join(directory_loc, jobname)

    params.update({f"jobname_{suffix}": f"{jobname}_{suffix}" for suffix in ["it", "e", "t", "c"]})

    first_transient_check = int(params["initial_FTTs"]) * int(params["FTT"])
    params.update({"transient_restart": f"{first_transient_check}"})

    fiveFTT = 5 * int(params["FTT"])
    params.update({"fiveFTT": f"{fiveFTT}"})

    dt_traj = 0.2
    traj_start = int(4 * int(params["FTT"]) + 400 * dt_traj)
    params.update({"traj_start": f"{traj_start}"})

    u = 0.2366431913
    mu = u / params["Re"]
    params.update({"u": f"{u}", "mu": f"{mu:.9E}"})

    params['a_plus_4']   = params['a'] + 4
    params['a_plus_6p5'] = params['a'] + 6.5
    params['a_plus_9p5'] = params['a'] + 9.5

    # Create the folder structure
    directory_path = Path(directory_string)

    if directory_path.exists() and not params.get("overwrite", False):
        print(f"Error: Directory {directory_string} already exists. Aborting to prevent overwrite.")
        return False

    (directory_path / "solution").mkdir(parents=True, exist_ok=True)
    (directory_path / "solution" / "trajectory").mkdir(parents=True, exist_ok=True)
    (directory_path / "partitiondir").mkdir(parents=True, exist_ok=True)
    Path(f"<trajectories_dir>/{jobname}").mkdir(parents=True, exist_ok=True)

    # Place transient_check.py inside the solution folder
    shutil.copy(Path("transient_check.py"), directory_path / "solution" / "transient_check.py")

    # Create .ini and .slurm files from templates
    for filename in ["transient", "extend", "trajectory"]:
        ini_content   = Path(f"{filename}_ini.txt").read_text().format(**params)
        slurm_content = Path(f"{filename}_slurm.txt").read_text().format(**params)
        (directory_path / f"{filename}.ini").write_text(ini_content)
        (directory_path / f"{filename}.slurm").write_text(slurm_content)

    # Create conversion files
    py_content    = Path("convert_to_h5_py.txt").read_text().format(**params)
    slurm_content = Path("convert_slurm.txt").read_text().format(**params)
    (directory_path / "convert_to_h5.py").write_text(py_content)
    (directory_path / "convert.slurm").write_text(slurm_content)

    # Generate the .geo file
    params.update({"geo_path": f"{directory_string}/cylinder.geo"})
    geo.main(
        Re=params["Re"], inflow_angle=params["inflow_angle"],
        a=params["a"], b=params["b"], n=params["n"],
        GR_BL=params["GR_BL"], t_BL=params["t_BL"],
        geo_output=params["geo_path"]
    )
    print(f"Project {directory_string} initialized with ini, geo, and slurm files.")

    # Meshing
    params.update({"msh_path": f"{directory_string}/cylinder.msh"})
    meshing_utils.mshgen(
        execute=params["meshing"],
        geo_path=params["geo_path"],
        msh_path=params["msh_path"]
    )

    # Importing
    params.update({"pyfrm_path": f"{directory_string}/cylinder.pyfrm"})
    meshing_utils.pyfrmgen(
        execute=params["importing"],
        msh_path=params["msh_path"],
        pyfrm_path=params["pyfrm_path"]
    )

    # Partitioning
    params.update({"partition_path": f"{directory_string}/partitiondir/"})
    meshing_utils.partitiongen(
        execute=params["partitioning"],
        pyfrm_path=params["pyfrm_path"],
        partition_path=params["partition_path"]
    )

    return True
