import os
import sys
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import marching_cubes

repo_src = Path(__file__).resolve().parents[1] / "src"
if str(repo_src) not in sys.path:
    sys.path.insert(0, str(repo_src))
from BZPlotter.coord import ReciprocalLattice


def read_bxsf(filename):
    """Read Fermi energy, reciprocal vectors, and band grids from a BXSF file."""
    data = {
        "fermi_energy": None,
        "band_data": [],
        "grid_dimensions": None,
        "origin": None,
        "vectors": None,
    }
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} not found.")

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


def _lattice_translations(lattice, nmax=2, include_zero=True):
    """Build reciprocal vectors n1*b1+n2*b2+n3*b3 in Cartesian coordinates."""
    nset = np.array(list(product(range(-nmax, nmax + 1), repeat=3)), dtype=float)
    if not include_zero:
        nset = nset[np.any(nset != 0.0, axis=1)]
    return lattice.frac_to_cart(nset)


def _ws_normals(lattice, nmax=2):
    """Return unique non-zero WS plane normals from reciprocal translations."""
    g = _lattice_translations(lattice, nmax=nmax, include_zero=False)
    g = g[np.linalg.norm(g, axis=1) > 1e-12]
    return np.unique(np.round(g, 12), axis=0)


def fold_to_wigner_seitz(kpts, lattice, nmax=2):
    """
    Fold k-points into first BZ by nearest reciprocal-lattice translation.
    """
    g_vectors = _lattice_translations(lattice, nmax=nmax, include_zero=True)
    disp = kpts[:, None, :] - g_vectors[None, :, :]
    idx = np.argmin(np.sum(disp**2, axis=2), axis=1)
    return kpts - g_vectors[idx]


def get_high_symmetry_points(lattice_type, lattice, points_file=None):
    """Return high-symmetry points in Cartesian coordinates."""
    if points_file:
        p = Path(points_file)
        if p.exists():
            points = {}
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    try:
                        frac = np.array(list(map(float, parts[:3])), dtype=float)
                    except ValueError:
                        continue
                    label = "G" if parts[3].upper() == "GAMMA" else parts[3]
                    points[label] = lattice.frac_to_cart(frac)
            if points:
                return points

    points_frac = {
        "cubic": {"G": [0, 0, 0], "X": [0.5, 0, 0], "M": [0.5, 0.5, 0], "R": [0.5, 0.5, 0.5]},
        "hexagonal": {"G": [0, 0, 0], "M": [0.5, 0, 0], "K": [1 / 3, 1 / 3, 0], "A": [0, 0, 0.5], "H": [1 / 3, 1 / 3, 0.5]},
        "tetragonal": {"G": [0, 0, 0], "X": [0.5, 0, 0], "M": [0.5, 0.5, 0], "Z": [0, 0, 0.5]},
    }
    selected = points_frac.get(lattice_type, points_frac["cubic"])
    return {label: lattice.frac_to_cart(np.asarray(coord, dtype=float)) for label, coord in selected.items()}


def _ws_vertex_mask(kpts, lattice, nmax=2, tol=1e-10):
    """
    Return mask of points inside first WS cell around Gamma:
    |k.G| <= |G|^2 / 2 for reciprocal vectors G in a finite shell.
    """
    g = _ws_normals(lattice, nmax=nmax)
    rhs = 0.5 * np.sum(g * g, axis=1) + tol
    return np.all(np.abs(kpts @ g.T) <= rhs[None, :], axis=1)


def _ws_polyhedron_edges(lattice, nmax=2, tol=1e-9):
    """
    Build WS-cell wireframe edges from half-space intersections.
    """
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


def draw_bz_boundary(ax, lattice, nmax=2):
    """Draw first Brillouin-zone boundary from WS construction."""
    for p0, p1 in _ws_polyhedron_edges(lattice, nmax=nmax):
        ax.plot3D(*zip(p0, p1), color="black", lw=1.3, alpha=0.8)


def plot_fermi_surface(data, band_idx, lattice_type="cubic", fold_ws=True, points_file=None):
    ef = data["fermi_energy"]
    grid = data["band_data"][band_idx]
    nx, ny, nz = data["grid_dimensions"]
    vecs = np.array(data["vectors"], dtype=float)
    lattice = ReciprocalLattice.from_B(vecs.T)

    verts, faces, _, _ = marching_cubes(grid, ef)

    frac = np.column_stack(
        [
            verts[:, 0] / (nx - 1),
            verts[:, 1] / (ny - 1),
            verts[:, 2] / (nz - 1),
        ]
    )

    verts_base = lattice.frac_to_cart(frac - 0.5)
    verts_plot = verts_base
    faces_plot = faces
    if fold_ws:
        # Fold after marching-cubes connectivity is built: one translation per triangle.
        tri = verts_base[faces]
        centers = tri.mean(axis=1)
        centers_folded = fold_to_wigner_seitz(centers, lattice, nmax=2)
        shifts = centers - centers_folded
        tri_folded = tri - shifts[:, None, :]

        # Keep only triangles fully inside WS after coherent face-wise folding.
        tri_points = tri_folded.reshape(-1, 3)
        in_ws = _ws_vertex_mask(tri_points, lattice, nmax=2).reshape(-1, 3)
        keep = np.all(in_ws, axis=1)
        tri_keep = tri_folded[keep]

        verts_plot = tri_keep.reshape(-1, 3)
        faces_plot = np.arange(len(verts_plot), dtype=int).reshape(-1, 3)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_trisurf(
        verts_plot[:, 0],
        verts_plot[:, 1],
        faces_plot,
        verts_plot[:, 2],
        cmap="Spectral",
        alpha=0.75,
        edgecolor="none",
    )

    draw_bz_boundary(ax, lattice, nmax=2)

    hs_points = get_high_symmetry_points(lattice_type, lattice, points_file=points_file)
    for label, coord in hs_points.items():
        ax.scatter(*coord, color="red", s=35)
        ax.text(*coord, label, size=10)

    max_dim = np.max(np.linalg.norm(vecs, axis=1)) * 0.7
    ax.quiver(0, 0, 0, max_dim, 0, 0, color="r", arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, max_dim, 0, color="g", arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, max_dim, color="b", arrow_length_ratio=0.1)

    ax.set_title(f"Band {band_idx} Fermi Surface\nLattice: {lattice_type.capitalize()} | Fold to WS: {fold_ws}")
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    plt.show()


if __name__ == "__main__":
    path = input("Path to .bxsf file: ").strip()
    lat_type = input("Lattice type (cubic/hexagonal/tetragonal): ").strip().lower() or "cubic"
    default_points = Path(path).resolve().parent / "HIGH_SYMMETRY_POINTS"
    msg = f"Path to HIGH_SYMMETRY_POINTS [{default_points}]: " if default_points.exists() else "Path to HIGH_SYMMETRY_POINTS [optional]: "
    points_input = input(msg).strip()
    points_file = points_input or (str(default_points) if default_points.exists() else None)

    bxsf_data = read_bxsf(path)
    print(f"Loaded {len(bxsf_data['band_data'])} bands.")

    band_input = int(input("Band index to plot: "))
    fold_input = input("Fold surface into first BZ? (Y/n): ").strip().lower()
    fold_ws = fold_input not in ("n", "no")

    plot_fermi_surface(bxsf_data, band_input, lat_type, fold_ws=fold_ws, points_file=points_file)
