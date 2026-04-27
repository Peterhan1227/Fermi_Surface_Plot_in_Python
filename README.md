# Fermi Surface Code

Tools for reading `*.bxsf` band-grid files, visualizing 3D Fermi surfaces, slicing those surfaces with arbitrary planes in reciprocal space, and converting closed orbit areas into de Haas-van Alphen (dHvA) frequencies.

This repository is aimed at an interactive research workflow:

- inspect a Fermi surface in 3D,
- choose a physically meaningful slice normal,
- sample a plane through the band structure,
- extract closed `E(k) = E_F` contours,
- report orbit areas and corresponding quantum-oscillation frequencies,
- optionally sweep many parallel planes to find extremal orbits.

The current maintained workflow lives in `Notebooks/` and uses a small reusable geometry package in `src/BZPlotter/`.

## What This Project Does

At a high level, the code solves a common condensed-matter workflow:

1. Read a BXSF file containing band energies on a periodic reciprocal-space grid.
2. Reconstruct the Fermi surface of a chosen band with marching cubes.
3. Build a plane perpendicular to a user-chosen direction (typically the magnetic-field direction).
4. Interpolate the 3D energy grid onto that 2D plane.
5. Find the `E = E_F` contours on the plane.
6. Measure enclosed areas of closed contours.
7. Convert those areas to dHvA frequencies through the Onsager relation.

The code is especially useful when you want a quick exploratory workflow before turning the procedure into a more formal analysis pipeline.

## Current State Of The Repo

- `Notebooks/FS_plot.py` is the main interactive 3D viewer.
- `Notebooks/slicer_f_one_plane.py` is the main single-plane slicing script.
- `Notebooks/slicer_f.py` is the main sweep script for extremal orbits.
- `src/BZPlotter/` contains the reusable reciprocal-space geometry helpers.
- `Notebooks/Archive/` contains older or superseded scripts kept for reference.
- A root-level `fermi_bxsf.py` is also present in this workspace as an older standalone plotter; the newer `Notebooks/FS_plot.py` is the better starting point.

The repository is script-driven rather than package-first: most end-user workflows happen through interactive prompts, not a polished command-line interface.

## Repository Layout

```text
Fermi-surface-code/
|-- Data/
|   |-- Cu.bxsf
|   |-- FERMISURFACE.bxsf
|   |-- Cu_SYMMETRY_PTS
|   `-- HIGH_SYMMETRY_POINTS
|-- Notebooks/
|   |-- FS_plot.py
|   |-- slicer_f.py
|   |-- slicer_f_one_plane.py
|   `-- Archive/
|       `-- slicer_fix_plane.py
|-- src/
|   `-- BZPlotter/
|       |-- __init__.py
|       |-- coord.py
|       |-- linalg.py
|       `-- plane.py
|-- fermi_bxsf.py
|-- pyproject.toml
`-- README.md
```

## Requirements

### Python Version

- Python `>= 3.10`

### Runtime Dependencies

The scripts in this repo use:

- `numpy`
- `matplotlib`
- `scikit-image`

`pyproject.toml` currently declares only `numpy`, so if you want the plotting and surface-extraction scripts to work, install the plotting dependencies manually as well.

### Installation

From the repository root:

```powershell
pip install numpy matplotlib scikit-image
```

If you want the `BZPlotter` package importable in editable mode:

```powershell
pip install -e .
```

That editable install is mainly useful for the geometry helpers in `src/BZPlotter/`. The BXSF parsing and slicing workflows are still primarily exposed as scripts in `Notebooks/`.

## Input Data Expectations

The code expects a BXSF file with:

- a `Fermi Energy:` line,
- a `BEGIN_BANDGRID_3D` section,
- grid dimensions,
- a reciprocal-space origin,
- three reciprocal vectors,
- one or more `BAND:` blocks containing scalar energy values on the grid.

The parser stores:

- `fermi_energy`
- `band_data`
- `grid_dimensions`
- `origin`
- `vectors`

Two details matter for interpretation:

- `origin` is used when mapping Cartesian sample points back into the periodic grid.
- The grid may or may not repeat the first plane at the last index; `Notebooks/slicer_f.py` includes logic to detect the true periodic length along each axis.

## Bundled Example Data

The `Data/` directory includes example inputs:

- `Data/Cu.bxsf`
- `Data/FERMISURFACE.bxsf`
- `Data/Cu_SYMMETRY_PTS`
- `Data/HIGH_SYMMETRY_POINTS`

The symmetry-point files list fractional reciprocal coordinates with labels and can be used to annotate 3D plots.

## Quick Start

Run all commands from the repository root so the scripts can reliably add `src/` to `sys.path`.

### 1. Plot A 3D Fermi Surface

```powershell
python .\Notebooks\FS_plot.py
```

You will be prompted for:

- path to the `.bxsf` file,
- lattice type (`cubic`, `hexagonal`, or `tetragonal`),
- optional high-symmetry point file,
- band index,
- whether to fold the surface into the first Wigner-Seitz Brillouin zone,
- display wrapping mode,
- optional manual fractional shift.

What the script does:

- reads the selected band grid,
- extracts the `E = E_F` isosurface with marching cubes,
- maps the surface into reciprocal-space Cartesian coordinates,
- optionally folds triangles into the first Wigner-Seitz cell,
- overlays a Brillouin-zone wireframe and high-symmetry points.

This is usually the first script to run when deciding what slice normal you want to test later.

### 2. Slice A Single Plane And Report dHvA Frequencies

```powershell
python .\Notebooks\slicer_f_one_plane.py
```

You will be prompted for:

- path to the `.bxsf` file,
- band index,
- whether to preview the 3D Fermi surface,
- base plane center in fractional reciprocal coordinates,
- plane normal in Cartesian reciprocal-space coordinates,
- offset along that normal,
- optional in-plane orientation hint,
- in-plane half-ranges,
- grid resolution,
- minimum area cutoff for keeping closed contours.

What the script returns:

- a contour plot of `E = E_F` on the selected plane,
- the plane center in both fractional and Cartesian coordinates,
- the number of closed and open contour segments,
- every retained closed contour area in `A^-2`,
- the corresponding frequency in tesla and kilotesla.

Use this when you already know which plane you want and want a detailed orbit-by-orbit report on that slice.

### 3. Sweep Parallel Planes To Find Extremal Orbits

```powershell
python .\Notebooks\slicer_f.py
```

This uses the same plane definition as the single-plane script, but adds a sweep:

- `t_min`
- `t_max`
- `n_slices`

The script evaluates a family of planes

```text
center_cart(t0) = center_cart(0) + t0 * n_hat
```

and reports:

- the maximum closed-orbit area found in the sweep,
- the minimum non-zero closed-orbit area,
- the corresponding slice offsets,
- slice centers in fractional and Cartesian coordinates,
- all closed-orbit areas on the best slices,
- all corresponding frequencies,
- a plot of max contour area versus offset,
- contour plots for the best maximum and minimum slices.

It also computes the Wigner-Seitz `t` window along the chosen normal and prints a warning if your sweep range does not cover the full first-zone interval.

This is the script to use when you are looking for extremal orbits for a chosen field direction.

## Typical Research Workflow

1. Run `Notebooks/FS_plot.py` to inspect the Fermi surface geometry.
2. Choose a likely field direction and define a slice normal.
3. Run `Notebooks/slicer_f_one_plane.py` to test one plane and verify that the contour you care about is being sampled correctly.
4. Run `Notebooks/slicer_f.py` to sweep along that normal and locate extremal closed orbits.
5. Compare the maximum or minimum orbit frequency against experiment or other calculations.

## Script Reference

### `Notebooks/FS_plot.py`

Primary role:

- interactive 3D Fermi-surface visualization.

Important implementation details:

- uses `skimage.measure.marching_cubes`,
- supports coherent triangle wrapping to avoid seam-spanning faces,
- can fold the displayed surface into the first Wigner-Seitz cell,
- can label high-symmetry points either from a file or from built-in defaults for simple lattice types.

Best for:

- choosing slice normals,
- checking whether the chosen band intersects the Fermi level,
- visually verifying whether a pocket is centered at Gamma, a zone corner, or a translated image.

### `Notebooks/slicer_f_one_plane.py`

Primary role:

- evaluate one user-defined plane and report every closed orbit on that slice.

Important implementation details:

- imports its core slicing logic from `Notebooks/slicer_f.py`,
- converts a fractional center and Cartesian normal into a concrete plane,
- samples the energy field on a rectangular plane grid,
- extracts contours at `E = E_F`,
- discards open contours and tiny closed contours below `area_tol`.

Best for:

- detailed inspection of one slice,
- comparing several pockets on the same plane,
- verifying whether a contour is physically closed before running a large sweep.

### `Notebooks/slicer_f.py`

Primary role:

- sweep many parallel planes and identify extremal orbit areas.

Important implementation details:

- detects periodic grid lengths even when a BXSF file duplicates endpoint planes,
- performs periodic trilinear interpolation,
- computes the Wigner-Seitz cross-section window along a normal,
- records both the largest and smallest non-zero closed contours encountered.

Best for:

- estimating extremal dHvA frequencies,
- checking how sensitive an orbit is to displacement along the field direction,
- identifying whether the center plane is already extremal.

### `src/BZPlotter/coord.py`

Contains the `ReciprocalLattice` class used for:

- fractional-to-Cartesian reciprocal-space transforms,
- Cartesian-to-fractional transforms,
- wrapping fractional coordinates into a canonical unit cell.

This is the coordinate backbone of the whole project.

### `src/BZPlotter/linalg.py`

Contains small vector helpers:

- `normalize`
- `orthonormal_plane_basis`

These are used to build stable in-plane basis vectors for slicing.

### `src/BZPlotter/plane.py`

Contains:

- `PlaneSpec`
- `plane_grid`

These turn a plane definition into a structured grid of Cartesian `k` points suitable for interpolation.

## Legacy And Archived Files

### `fermi_bxsf.py`

This is an older standalone Fermi-surface plotting script at the repository root. It overlaps conceptually with `Notebooks/FS_plot.py` but uses an older plotting path and a simpler Brillouin-zone treatment.

Recommendation:

- use it only if you specifically want to compare against older behavior,
- otherwise start with `Notebooks/FS_plot.py`.

### `Notebooks/Archive/slicer_fix_plane.py`

This is an older single-plane slicing workflow retained for reference. The newer `Notebooks/slicer_f_one_plane.py` is the cleaner and more consistent single-plane entry point.

## Units And Interpretation

The code assumes reciprocal-space Cartesian coordinates are in inverse angstroms, so:

- contour areas are reported in `A^-2`,
- dHvA frequencies are reported in `T` and `kT`.

The area-to-frequency conversion used in `Notebooks/slicer_f.py` is:

```text
F = (hbar / 2 pi e) * A
```

with the area converted from `A^-2` to `m^-2` internally.

Important geometric conventions:

- `center_frac` is a point in fractional reciprocal coordinates relative to the reciprocal basis vectors.
- `normal_cart` is a Cartesian direction in reciprocal space.
- `offset_along_normal` and sweep parameter `t0` are distances along the normalized Cartesian normal.
- Changing `center_frac` changes the physical plane being sampled; it is not just a display translation.

## Common Pitfalls

### Running From The Wrong Directory

These scripts are easiest to run from the repository root. They try to add `src/` to `sys.path`, but running them from unusual working directories is still the most common import-related failure mode.

### Empty Contours

If you get no closed `E = E_F` contour on a slice:

- the band may not cross the Fermi level on that plane,
- the plane patch may be too small,
- the grid resolution may be too coarse,
- the selected normal may miss the pocket entirely.

Try increasing:

- `half_range`,
- `shape`,
- sweep span in `t`,
- or checking the band with `FS_plot.py` first.

### Open Instead Of Closed Contours

An open contour usually means one of the following:

- the orbit genuinely leaves the sampled rectangular patch,
- the patch is too small,
- the selected plane intersects an extended sheet rather than a closed pocket.

Open contours are not treated as valid dHvA orbits by the scripts.

### Sweep Range Too Narrow

`Notebooks/slicer_f.py` prints the Wigner-Seitz `t` window along the selected normal. If your chosen `[t_min, t_max]` does not span that full interval, you may miss the true extremal orbit inside the first zone.

### Dependency Confusion

If `numpy` is installed but the 3D plot fails, the missing package is often `scikit-image`. If plotting fails entirely, check `matplotlib`.

## Programmatic Reuse

The reusable package in `src/BZPlotter/` currently focuses on geometry rather than the full BXSF workflow. In other words:

- coordinate transforms are packaged cleanly,
- plane-grid generation is packaged cleanly,
- the end-to-end BXSF reading, interpolation, contouring, and sweep logic still lives in the scripts.

So if you want to build a larger codebase on top of this repo, the cleanest starting point is usually:

1. import `BZPlotter` for geometry utilities,
2. copy or refactor the relevant functions from `Notebooks/slicer_f.py`,
3. turn the interactive prompt flow into explicit function arguments or a real CLI.

## Limitations

- No automated test suite is currently included.
- The main user interfaces are interactive prompts rather than command-line flags.
- Packaging metadata is minimal compared with actual runtime dependencies.
- The active workflow is split across notebook-style scripts instead of a single importable analysis package.
- Some older scripts remain in the repo for historical comparison, so not every file reflects the latest implementation style.

## Suggested First Run

If you want one practical starting point with the bundled data:

1. run `python .\Notebooks\FS_plot.py`,
2. give it `Data\Cu.bxsf`,
3. use `Data\Cu_SYMMETRY_PTS` when asked for symmetry labels,
4. inspect a band that crosses `E_F`,
5. then move to `python .\Notebooks\slicer_f_one_plane.py` with the same file.

That path gives you the fastest feel for how the repository is meant to be used.

## License

No license file is currently included in the repository.
