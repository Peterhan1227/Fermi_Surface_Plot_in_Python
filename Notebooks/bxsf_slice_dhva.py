"""BXSF -> Fermi-surface plot + planar cross-section area -> dHvA frequency.

Workflow:
- Read .bxsf (VASPKIT) band energy grids
- (Optional) Plot 3D Fermi surface (marching cubes)
- Define a plane (normal ~ B-field direction)
- Interpolate E(k) from the 3D grid onto a 2D plane grid
- Extract E=E_F contours, compute enclosed areas, convert to dHvA frequencies
"""

from __future__ import annotations

import os
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import marching_cubes

# --- imports (work in your repo OR in this sandbox) ---
try:
    from BZPlotter.coord import ReciprocalLattice  # your repo style
except Exception:  # pragma: no cover
    try:
        from coord import ReciprocalLattice  # local sandbox style
    except Exception as e:
        raise ImportError("Could not import ReciprocalLattice (BZPlotter.coord or coord)") from e

try:
    from bzplotter.linalg import orthonormal_plane_basis  # your repo style
except Exception:  # pragma: no cover
    try:
        from linalg import orthonormal_plane_basis  # local sandbox style
    except Exception as e:
        raise ImportError("Could not import orthonormal_plane_basis (bzplotter.linalg or linalg)") from e


# -------------------------
# Reading BXSF
# -------------------------

def read_bxsf(filename: str) -> dict:
    data = {
        "fermi_energy": None,
        "band_data": [],
        "grid_dimensions": None,
        "origin": None,
        "vectors": None,
    }
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    i = 0
    while i < len(lines):
        if "Fermi Energy:" in lines[i]:
            data["fermi_energy"] = float(lines[i].split(":")[1].strip())
        elif "BEGIN_BANDGRID_3D" in lines[i]:
            i += 1
            n_bands = int(lines[i])
            i += 1
            data["grid_dimensions"] = list(map(int, lines[i].split()))
            i += 1
            data["origin"] = np.array(list(map(float, lines[i].split())), dtype=float)
            i += 1
            data["vectors"] = [list(map(float, lines[i + j].split())) for j in range(3)]
            i += 3

            total = int(np.prod(data["grid_dimensions"]))
            for _ in range(n_bands):
                while i < len(lines) and not lines[i].startswith("BAND:"):
                    i += 1
                i += 1
                pts = []
                while len(pts) < total and i < len(lines):
                    pts.extend(map(float, lines[i].split()))
                    i += 1
                data["band_data"].append(np.array(pts).reshape(*data["grid_dimensions"]))
        i += 1

    return data


# -------------------------
# WS BZ wireframe helpers (from your fermi_bxsf_3)
# -------------------------

def _lattice_translations(lattice: ReciprocalLattice, nmax=2, include_zero=True):
    nset = np.array(list(product(range(-nmax, nmax + 1), repeat=3)), dtype=float)
    if not include_zero:
        nset = nset[np.any(nset != 0.0, axis=1)]
    return lattice.frac_to_cart(nset)


def _ws_normals(lattice: ReciprocalLattice, nmax=2):
    g = _lattice_translations(lattice, nmax=nmax, include_zero=False)
    g = g[np.linalg.norm(g, axis=1) > 1e-12]
    return np.unique(np.round(g, 12), axis=0)


def _ws_polyhedron_edges(lattice: ReciprocalLattice, nmax=2, tol=1e-9):
    normals = _ws_normals(lattice, nmax=nmax)
    bvals = 0.5 * np.sum(normals * normals, axis=1)

    vertices = []
    for i, j, k in combinations(range(len(normals)), 3):
        a = np.vstack([normals[i], normals[j], normals[k]])
        if abs(np.linalg.det(a)) < 1e-10:
            continue
        x = np.linalg.solve(a, np.array([bvals[i], bvals[j], bvals[k]]))
        if np.all(normals @ x <= bvals + tol):
            vertices.append(x)

    if not vertices:
        return []

    verts = np.unique(np.round(np.asarray(vertices), 10), axis=0)
    active = np.isclose(normals @ verts.T, bvals[:, None], atol=5e-7, rtol=0.0)

    edges = []
    n_verts = len(verts)
    for i in range(n_verts):
        for j in range(i + 1, n_verts):
            if np.count_nonzero(active[:, i] & active[:, j]) >= 2:
                if np.linalg.norm(verts[i] - verts[j]) > 1e-8:
                    edges.append((verts[i], verts[j]))
    return edges


def draw_bz_boundary(ax, lattice: ReciprocalLattice, nmax=2):
    for p0, p1 in _ws_polyhedron_edges(lattice, nmax=nmax):
        ax.plot3D(*zip(p0, p1), color="black", lw=1.3, alpha=0.8)


# -------------------------
# 3D Fermi surface plot
# -------------------------

def plot_fermi_surface(data: dict, band_idx: int, fold_ws: bool = True):
    ef = data["fermi_energy"]
    grid = data["band_data"][band_idx]
    nx, ny, nz = data["grid_dimensions"]
    vecs = np.array(data["vectors"], dtype=float)
    lattice = ReciprocalLattice.from_B(vecs.T)

    verts, faces, _, _ = marching_cubes(grid, ef)

    frac = np.column_stack([
        verts[:, 0] / (nx - 1),
        verts[:, 1] / (ny - 1),
        verts[:, 2] / (nz - 1),
    ])

    # BXSF grids are typically centered; keep the same convention as your original code.
    verts_cart = lattice.frac_to_cart(frac - 0.5)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_trisurf(
        verts_cart[:, 0],
        verts_cart[:, 1],
        faces,
        verts_cart[:, 2],
        cmap="Spectral",
        alpha=0.75,
        edgecolor="none",
    )

    draw_bz_boundary(ax, lattice, nmax=2)
    ax.set_title(f"Band {band_idx} Fermi Surface")
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    plt.show()


# -------------------------
# Plane grid + trilinear interpolation on periodic BXSF grid
# -------------------------

def plane_grid(center_cart: np.ndarray,
               normal_cart: np.ndarray,
               orient_hint_cart: np.ndarray,
               shape=(401, 401),
               half_range=(1.0, 1.0)):
    """Return pts(N,3), S(Ny,Nx), T(Ny,Nx), and basis u,v,n (all in Cartesian)."""
    Nx, Ny = shape
    rx, ry = half_range

    u, v, n = orthonormal_plane_basis(normal=normal_cart, orient_hint=orient_hint_cart)

    s = np.linspace(-rx, rx, Nx)
    t = np.linspace(-ry, ry, Ny)
    S, T = np.meshgrid(s, t, indexing="xy")

    pts_full = center_cart + S[..., None] * u + T[..., None] * v
    pts = pts_full.reshape(-1, 3)

    return pts, S, T, u, v, n


def interp_trilinear_periodic(grid: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Trilinear interpolation for points in index coordinates (x,y,z).

    Assumes BXSF-style periodic grid where last index repeats the first, so the
    true period is (Nx-1, Ny-1, Nz-1).
    """
    nx, ny, nz = grid.shape
    px, py, pz = nx - 1, ny - 1, nz - 1

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    i0 = np.floor(x).astype(int) % px
    j0 = np.floor(y).astype(int) % py
    k0 = np.floor(z).astype(int) % pz

    i1 = (i0 + 1) % px
    j1 = (j0 + 1) % py
    k1 = (k0 + 1) % pz

    tx = x - np.floor(x)
    ty = y - np.floor(y)
    tz = z - np.floor(z)

    c000 = grid[i0, j0, k0]
    c100 = grid[i1, j0, k0]
    c010 = grid[i0, j1, k0]
    c110 = grid[i1, j1, k0]
    c001 = grid[i0, j0, k1]
    c101 = grid[i1, j0, k1]
    c011 = grid[i0, j1, k1]
    c111 = grid[i1, j1, k1]

    c00 = c000 * (1 - tx) + c100 * tx
    c10 = c010 * (1 - tx) + c110 * tx
    c01 = c001 * (1 - tx) + c101 * tx
    c11 = c011 * (1 - tx) + c111 * tx

    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty

    return c0 * (1 - tz) + c1 * tz


def energies_on_plane_from_bxsf(
    data: dict,
    band_idx: int,
    center_cart: np.ndarray,
    normal_cart: np.ndarray,
    orient_hint_cart: np.ndarray,
    *,
    shape=(401, 401),
    half_range=(1.0, 1.0),
):
    """Return (S,T,E2d,u,v,n,lattice) where E2d has shape (Ny,Nx)."""
    grid = data["band_data"][band_idx]
    nx, ny, nz = data["grid_dimensions"]
    vecs = np.array(data["vectors"], dtype=float)
    lattice = ReciprocalLattice.from_B(vecs.T)

    pts, S, T, u, v, n = plane_grid(
        center_cart=center_cart,
        normal_cart=normal_cart,
        orient_hint_cart=orient_hint_cart,
        shape=shape,
        half_range=half_range,
    )

    # Cartesian k -> fractional reciprocal coordinates
    # Keep the same BXSF centering convention as the FS plot: frac in [-0.5,0.5)
    frac = lattice.cart_to_frac(pts) + 0.5
    frac = frac - np.floor(frac)  # wrap to [0,1)

    xyz = np.column_stack([
        frac[:, 0] * (nx - 1),
        frac[:, 1] * (ny - 1),
        frac[:, 2] * (nz - 1),
    ])

    e = interp_trilinear_periodic(grid, xyz)
    E2d = e.reshape(S.shape)
    return S, T, E2d, u, v, n, lattice


# -------------------------
# Area + frequency
# -------------------------

def polygon_area(xy: np.ndarray) -> float:
    """Shoelace area for a closed polygon given by points (N,2)."""
    if len(xy) < 3:
        return 0.0
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def contour_areas(S: np.ndarray, T: np.ndarray, E2d: np.ndarray, level: float):
    """Return list of areas (in plane coords, i.e. Å^-2) for closed E=level contours."""
    cs = plt.contour(S, T, E2d, levels=[level])
    areas = []
    for coll in cs.collections:
        for p in coll.get_paths():
            v = p.vertices
            if len(v) < 3:
                continue
            # Close if needed
            if np.linalg.norm(v[0] - v[-1]) > 1e-10:
                v = np.vstack([v, v[0]])
            areas.append(polygon_area(v[:-1]))
    plt.close()
    return areas


def area_to_frequency_T(area_Ainv2: float) -> float:
    """Onsager relation: F = (ħ/2πe) A.

    area_Ainv2 is in Å^-2. Convert to m^-2 via (1 Å^-1 = 1e10 m^-1).
    """
    hbar = 1.054_571_817e-34  # J*s
    e = 1.602_176_634e-19     # C
    area_m2inv = area_Ainv2 * 1e20
    return (hbar / (2 * np.pi * e)) * area_m2inv


# -------------------------
# User-facing workflow
# -------------------------

def slice_and_dhva(
    data: dict,
    band_idx: int,
    *,
    center_frac=(0.0, 0.0, 0.0),
    normal_cart=(0.0, 0.0, 1.0),
    orient_hint_cart=(1.0, 0.0, 0.0),
    shape=(401, 401),
    half_range=(1.0, 1.0),
    show=True,
):
    ef = float(data["fermi_energy"])
    vecs = np.array(data["vectors"], dtype=float)
    lattice = ReciprocalLattice.from_B(vecs.T)

    center_cart = lattice.frac_to_cart(np.asarray(center_frac, dtype=float))

    S, T, E2d, u, v, n, _ = energies_on_plane_from_bxsf(
        data,
        band_idx,
        center_cart=center_cart,
        normal_cart=np.asarray(normal_cart, dtype=float),
        orient_hint_cart=np.asarray(orient_hint_cart, dtype=float),
        shape=shape,
        half_range=half_range,
    )

    areas = contour_areas(S, T, E2d, ef)
    areas = sorted([a for a in areas if a > 0], reverse=True)
    freqs = [area_to_frequency_T(a) for a in areas]

    if show:
        plt.figure(figsize=(7, 6))
        plt.contour(S, T, E2d, levels=[ef])
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title(f"Band {band_idx} slice: E=E_F contours\nnormal ~ {np.asarray(normal_cart)}")
        plt.xlabel("s (Å$^{-1}$)")
        plt.ylabel("t (Å$^{-1}$)")
        plt.show()

        if areas:
            a0 = areas[0]
            f0 = freqs[0]
            print(f"Contours found: {len(areas)}")
            print(f"Largest area: {a0:.6g} Å^-2")
            print(f"Frequency:    {f0:.6g} T  ({f0/1e3:.6g} kT)")
        else:
            print("No closed E=E_F contours found on this slice.")

    return areas, freqs


if __name__ == "__main__":
    path = input("Path to .bxsf file: ").strip()
    data = read_bxsf(path)
    print(f"Loaded {len(data['band_data'])} bands. E_F = {data['fermi_energy']}")

    band = int(input("Band index: ").strip())

    do3d = input("Plot 3D Fermi surface first? (y/N): ").strip().lower() in ("y", "yes")
    if do3d:
        plot_fermi_surface(data, band)

    # Plane inputs (keep it simple, defaults work)
    c_in = input("Center (fractional b1 b2 b3) [0 0 0]: ").strip()
    center_frac = tuple(map(float, c_in.split())) if c_in else (0.0, 0.0, 0.0)

    n_in = input("Normal (Cartesian) [0 0 1]: ").strip()
    normal_cart = tuple(map(float, n_in.split())) if n_in else (0.0, 0.0, 1.0)

    o_in = input("Orient hint (Cartesian) [1 0 0]: ").strip()
    orient_hint = tuple(map(float, o_in.split())) if o_in else (1.0, 0.0, 0.0)

    r_in = input("Half-range rx ry (Å^-1) [1 1]: ").strip()
    half_range = tuple(map(float, r_in.split())) if r_in else (1.0, 1.0)

    s_in = input("Grid Nx Ny [401 401]: ").strip()
    shape = tuple(map(int, s_in.split())) if s_in else (401, 401)

    slice_and_dhva(
        data,
        band,
        center_frac=center_frac,
        normal_cart=normal_cart,
        orient_hint_cart=orient_hint,
        half_range=half_range,
        shape=shape,
        show=True,
    )
