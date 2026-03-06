# Fermi Surface Code

Utilities for reading `.bxsf` band-grid files, plotting 3D Fermi surfaces in the first Brillouin zone, and extracting dHvA frequencies from planar slices.

This README reflects the cleaned workflow scripts in `Notebooks/`:
- `FS_plot.py`
- `slicer_fix_plane.py`
- `slicer_f.py`

## Project Layout

- `Notebooks/FS_plot.py`
  - Interactive 3D Fermi-surface viewer.
  - Marching-cubes isosurface at `E = E_F`.
  - Optional coherent wrapping and WS folding into the first BZ.
  - Optional high-symmetry point labels from file.
- `Notebooks/slicer_fix_plane.py`
  - Single-plane slicing workflow.
  - Interpolates `E(k)` on a user-defined plane and computes contour areas/frequencies.
- `Notebooks/slicer_f.py`
  - Includes all single-plane logic plus parallel-plane sweep.
  - Finds the maximum closed orbit area and reports the corresponding dHvA frequency.
- `src/BZPlotter/`
  - Core helpers (`coord.py`, `plane.py`, `linalg.py`) used by notebook scripts.
- `Data/`
  - Example inputs (`Cu.bxsf`, `FERMISURFACE.bxsf`) and symmetry-point files.

## Requirements

- Python `>=3.10`
- `numpy`
- `matplotlib`
- `scikit-image`

Install runtime dependencies:

```powershell
pip install numpy matplotlib scikit-image
```

Optional editable install (if you want to import `BZPlotter` outside these scripts):

```powershell
pip install -e .
```

## Quick Start

Run commands from repository root.

### 1) 3D Fermi Surface Plot

```powershell
python .\Notebooks\FS_plot.py
```

You will be prompted for:
- path to `.bxsf`
- lattice type (`cubic/hexagonal/tetragonal`)
- optional `HIGH_SYMMETRY_POINTS` file path
- band index
- whether to fold to first WS BZ
- wrap mode (`gamma/corner/none`)
- optional manual fractional shift

### 2) Single Plane Slice and dHvA

```powershell
python .\Notebooks\slicer_fix_plane.py
```

You will be prompted for:
- `.bxsf` path and band index
- optional 3D FS preview
- plane center in fractional coordinates
- plane normal in Cartesian coordinates
- optional orient hint (blank = auto)
- in-plane half-range and grid resolution

Output:
- contour plot of `E = E_F` in plane coordinates
- closed contour areas (`A^-2`) and dHvA frequencies (`T`, `kT`)

### 3) Sweep Slices to Find Maximum Orbit

```powershell
python .\Notebooks\slicer_f.py
```

Same inputs as single-plane mode, plus sweep parameters:
- `t_min t_max n_slices`

Output:
- `max area vs t0` plot
- best-slice contour plot
- best `t0`, center (frac/cart), max area, frequency, contour count
- center-plane (`t0 = 0`) summary for direct comparison

## Data Notes

- BXSF `origin` is used during interpolation to map Cartesian points into the correct periodic grid cell.
- Changing `center_frac` does not "move coordinates"; it changes which physical plane is sampled.
- If comparing single-plane and sweep results, keep `shape`, `half_range`, and normal/center conventions identical.

## Typical Workflow

1. Use `FS_plot.py` to inspect the pocket geometry and choose a physically meaningful normal.
2. Use `slicer_fix_plane.py` to test one specific slice.
3. Use `slicer_f.py` to scan offsets along that normal and locate the extremal orbit.

## Troubleshooting

- `ImportError: Could not import ReciprocalLattice/PlaneSpec...`
  - Run from repo root so scripts can add `src/` to `sys.path`.
- Empty or tiny contours:
  - Increase `half_range` and/or grid size.
  - Verify chosen band intersects `E_F`.
- Sweep maximum lower than expected from a known slice:
  - Ensure your expected slice (`t0=0`) is inside sweep range and sampled.
  - Use odd `n_slices` or rely on the built-in forced `t0=0` evaluation in `slicer_f.py`.

## License

No license file is currently included.
