# Fermi Surface Plot in Python

Simple tools for reading `.bxsf` files, plotting 3D Fermi surfaces, and estimating dHvA frequencies from planar slices.

## What This Repo Contains

- `Notebooks/fermi_bxsf_3.py`
  - Main 3D Fermi-surface plotting script
  - Reads BXSF, builds Fermi isosurface with marching cubes, draws first BZ boundary
- `Notebooks/bxsf_slice_dhva_refactor.py`
  - BXSF slicing workflow
  - Interpolates energies on a plane, extracts `E = E_F` contours, computes area and dHvA frequency
- `src/BZPlotter/`
  - Reusable coordinate, plane, and linear algebra helpers
- `Data/`
  - Example input data (e.g., `FERMISURFACE.bxsf`)

## Requirements

Python 3.10+

Install dependencies:

```powershell
pip install numpy matplotlib scikit-image
```

## Quick Start

From repo root:

### 1) Plot a 3D Fermi surface

```powershell
python .\Notebooks\fermi_bxsf_3.py
```

Then follow prompts (path to `.bxsf`, lattice type, band index, etc.).

### 2) Slice and estimate dHvA frequency

```powershell
python .\Notebooks\bxsf_slice_dhva_refactor.py
```

Then enter:
- BXSF file path
- band index
- slice center/normal
- interpolation range and grid

## Notes

- Scripts assume reciprocal lattice vectors are read from BXSF and passed with the expected orientation for `ReciprocalLattice`.
- `orient_hint` in slicing can be left blank to auto-generate a perpendicular default.
- `__pycache__` files are local cache files and not part of core source logic.

## License

No license file is currently included. Add one if you plan to share or publish broadly.
