# Dataset Generation Pipeline

This repository contains the scripts used to generate the CFD dataset accompanying the paper. The pipeline produces high-fidelity Navier–Stokes solutions around superellipse-shaped cylinders across a range of flow conditions, and converts the raw solver output into structured HDF5 files suitable for machine learning.

---

## Overview

The dataset covers flow past cylinders whose cross-section is a **superellipse** (Lamé curve), parameterised by semi-axis `a`, fixed semi-axis `b = 0.5`, exponent `n`, and inflow angle `alpha`. Reynolds numbers range from 1000 to 3900. Parameter combinations are drawn using a **Sobol sequence** to ensure quasi-uniform coverage of the four-dimensional input space. Each sample is automatically assigned to a train / validation / test split (80 / 10 / 10).

Each simulation case proceeds through four sequential stages:

```
[1] Setup       →   [2] Transient     →   [3] Extend (optional)   →   [4] Trajectory
  geo + mesh          run to              further time-marching        collect snapshots
  + partition         statistical                                       for the dataset
                      stationarity
                      check
                                                                              ↓
                                                                         [5] Convert
                                                                         .pyfrs → HDF5
```

All simulations are run with **PyFR** (a high-order flux reconstruction solver) on GPU nodes. Mesh generation uses **Gmsh**. Post-processing uses **PyVista** and **h5py**.

---

## Parameter Space

| Parameter | Symbol | Range | Sampling |
|-----------|--------|-------|----------|
| Reynolds number | Re | 1000 – 3900 | Linear (integer) |
| Inflow angle | α | 0° – 45° | Linear (integer degrees) |
| Semi-axis (streamwise) | a | 1/6 – 1.5 | Log-uniform |
| Cross-section exponent | n | 1, 2, 5, ∞ | Discrete (uniform) |

`n = 1` gives a rhombus, `n = 2` an ellipse, `n = 5` a rounded rectangle, and `n = ∞` a rectangle. The fixed semi-axis `b = 0.5` sets the cross-stream half-width.

A scrambled Sobol sequence of 2¹⁰ = 1024 samples is generated once and saved to `sobol_samples.csv`. Cases are set up by consuming rows from this file in order, so adding more cases later simply continues from where the previous run left off.

---

## File Structure

### Setup scripts (run once per campaign)

| File | Purpose |
|------|---------|
| `driver.py` | Entry point. Generates the Sobol samples (optionally) and calls `setup_simulation` for the requested number of cases. |
| `driver50.slurm` | Example SLURM job script for running `driver.py` on a CPU node. |
| `sobol.py` | `SobolSampler` class: generates, transforms, saves, and plots the parameter samples. |
| `setup.py` | `setup_simulation()`: creates the directory structure and all input files for a single case. |
| `geo.py` | Generates a Gmsh `.geo` file for the given superellipse geometry and Reynolds number. Mesh sizes are interpolated by power law between two reference resolutions. |
| `meshing_utils.py` | Thin wrappers around Gmsh (mesh generation), `pyfr import` (format conversion), and `pyfr partition` (domain decomposition for 4 GPUs). |

### Per-case input file templates

Each simulation case gets its own directory named `Re{Re}_alpha{alpha}_a{a}_n{n}/`. Inside, `setup.py` writes the following files by filling placeholders into the templates below.

| Template | Output file | Purpose |
|----------|-------------|---------|
| `transient.ini` | `transient.ini` | PyFR config for the initial transient run (from t = 0 to `initial_FTTs × FTT`). Records wall forces and point-probe velocity samples for the stationarity check. Writes a solution snapshot every FTT time units. |
| `transient.slurm` | `transient.slurm` | SLURM job for the transient run. Calls `transient_check.py` on completion. |
| `extend.ini` | `extend.ini` | PyFR config for optional continuation runs. Automatically picks up the latest `.pyfrs` restart file and advances by a further 5 FTTs. |
| `extend.slurm` | `extend.slurm` | SLURM job for the extend run. |
| `trajectory.ini` | `trajectory.ini` | PyFR config for the data-collection phase. Writes high-frequency snapshots (every Δt = 0.2) to a separate trajectories directory, restricted to a bounding box around the cylinder. |
| `trajectory.slurm` | `trajectory.slurm` | SLURM job for the trajectory run. |
| `convert_to_h5.py` | `convert_to_h5.py` | Converts trajectory `.pyfrs` snapshots to HDF5 (see Stage 5 below). |
| `convert.slurm` | `convert.slurm` | SLURM job for the conversion step. |

### Analysis

| File | Purpose |
|------|---------|
| `transient_check.py` | Reads the point-probe CSV written during the transient/extend run and computes a convergence statistic χ for the u and v velocity signals at three downstream x-locations. χ measures how much the time-averaged statistics change between successive quarters of the signal. A low χ (< ~10%) indicates statistical stationarity. Saves a diagnostic plot. |

---

## Simulation Stages in Detail

### Stage 1 — Setup

Running `driver.py --setup_nr N` creates N case directories. For each case, `setup_simulation()`:

1. Derives a unique case name: `Re{Re}_alpha{alpha}_a{a}_n{n}`.
2. Computes physical parameters: the freestream velocity `u = 0.2366...` (Mach 0.2), dynamic viscosity `mu = u / Re`, and derived time scales.
3. Writes the directory tree: `solution/`, `solution/trajectory/`, `partitiondir/`.
4. Fills all `.ini` and `.slurm` templates with case-specific values and writes them to disk.
5. Generates the Gmsh `.geo` file via `geo.py`.
6. Optionally runs Gmsh to produce a `.msh` file, imports it into PyFR (`.pyfrm`), and partitions it into 4 subdomains.

### Stage 2 — Transient run

Submit `transient.slurm`. PyFR integrates the compressible Navier–Stokes equations (4th-order flux reconstruction, RK45 with PI step control) from a perturbed uniform initial condition up to `initial_FTTs × FTT` convective time units. Two plugins run in parallel:

- **fluidforce**: records lift and drag on the cylinder wall every 250 steps.
- **sampler**: records primitive variables at 64 equally-spaced spanwise points on three downstream rake lines (at x = a + 4, a + 6.5, a + 9.5) every 250 steps, writing to `transient_data.csv`.

On completion, `transient_check.py` is called automatically to assess stationarity.

### Stage 3 — Extend (if needed)

If the convergence statistic χ is still large, submit `extend.slurm` one or more times. The script automatically finds the latest `.pyfrs` restart file, updates `t0` and `tend` in `extend.ini`, and advances the simulation by a further 5 FTTs, continuing to accumulate the point-probe CSV.

### Stage 4 — Trajectory collection

Once stationary, submit `trajectory.slurm`. PyFR restarts from the latest solution and writes full-field snapshots at Δt = 0.2 for 80 time units, restricted to the box x ∈ [−3, 10], y ∈ [−3.5, 3.5], z ∈ [0, 2π]. These `.pyfrs` files are written to a separate trajectories directory.

### Stage 5 — Conversion to HDF5

Submit `convert.slurm`. `convert_to_h5.py`:

1. Sorts all `.pyfrs` trajectory files by simulation time.
2. For each snapshot, exports to a high-order `.vtu` file using `pyfr export` (subdivision level 4, single precision) and reads it with PyVista.
3. Applies a node-deduplication step (`clean`) to remove duplicate solution points at element boundaries.
4. On the first snapshot, saves mesh coordinates (x, y, z) to a separate `_coords.h5` file and a template `.vtu` for reconstruction. Periodically verifies that mesh coordinates have not shifted.
5. Appends solution fields (ρ, p, u, v, w) to batch HDF5 files in groups of 100 snapshots: `_b0_99.h5`, `_b100_199.h5`, etc. Each group is named `sol_t{index}`.
6. Deletes intermediate `.vtu` files to conserve disk space.

---

## Case Directory Layout (after full pipeline)

```
Re{Re}_alpha{alpha}_a{a}_n{n}/
├── cylinder.geo              # Gmsh geometry file
├── cylinder.msh              # Gmsh mesh
├── cylinder.pyfrm            # PyFR mesh (full)
├── partitiondir/
│   └── cylinder.pyfrm        # Partitioned PyFR mesh (4 parts)
├── transient.ini / .slurm
├── extend.ini / .slurm
├── trajectory.ini / .slurm
├── convert_to_h5.py / convert.slurm
└── solution/
    ├── transient_data.csv    # Point-probe time series (stationarity check)
    ├── cylinder-forces.csv   # Lift and drag time series
    ├── transient_check.py
    ├── {simname}_t*.pyfrs    # Solution snapshots (one per FTT)
    └── trajectory/           # (not used directly; snapshots go to trajectories dir)

<trajectories_dir>/{simname}/
└── {simname}_ts*.pyfrs       # High-frequency trajectory snapshots

<dataset_dir>/{tag}/{simname}/
├── {simname}_coords.h5       # Mesh node coordinates (x, y, z)
├── {simname}_template.vtu    # Mesh topology for reconstruction
└── {simname}_b{start}_{end}.h5   # Solution batches (rho, p, u, v, w)
```

---

## Key Constants and Time Scales

| Quantity | Value | Notes |
|----------|-------|-------|
| Freestream velocity | u = 0.2366431913 | Mach 0.2 with γ = 1.4 |
| Reference density | ρ = 1 | Non-dimensionalised |
| Reference pressure | p = 1 | Non-dimensionalised |
| Prandtl number | Pr = 0.72 | Air |
| Solver order | 4 | Flux reconstruction |
| FTT | 136 time units | One flow-through time (domain length / u) |
| Snapshot interval (trajectory) | Δt = 0.2 | |
| Boundary layer thickness | t_BL = 0.5 | Structured hex layer around object |
| BL growth rate | GR_BL = 1.021 | Geometric progression |
| Spanwise length | 2π | Periodic |
| Spanwise cells | 30 | |

---

## Dependencies

- [PyFR](https://www.pyfr.org/) — high-order CFD solver
- [Gmsh](https://gmsh.info/) — mesh generation (Python API)
- [PyVista](https://pyvista.org/) — VTK-based mesh I/O
- [h5py](https://www.h5py.org/) — HDF5 file I/O
- [Polars](https://pola.rs/) — fast DataFrame library (used in `transient_check.py`)
- [SciPy](https://scipy.org/) — Sobol sequence generation (`scipy.stats.qmc`)
- NumPy, Matplotlib, Pandas
