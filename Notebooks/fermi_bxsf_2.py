import os
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import marching_cubes


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


def fold_to_wigner_seitz(kpts, vecs):
    """
    Fold k-points into first BZ by nearest reciprocal-lattice translation.
    vecs shape: (3,3), rows are b1, b2, b3.
    """
    nset = np.array(list(product(range(-1, 2), repeat=3)), dtype=float)
    g_vectors = nset @ vecs
    folded = np.empty_like(kpts)

    for i, k in enumerate(kpts):
        d2 = np.sum((k - g_vectors) ** 2, axis=1)
        folded[i] = k - g_vectors[np.argmin(d2)]

    return folded


def get_high_symmetry_points(lattice_type, vecs):
    """Return simple high-symmetry points in Cartesian coordinates."""
    points_frac = {
        "cubic": {"G": [0, 0, 0], "X": [0.5, 0, 0], "M": [0.5, 0.5, 0], "R": [0.5, 0.5, 0.5]},
        "hexagonal": {"G": [0, 0, 0], "M": [0.5, 0, 0], "K": [1 / 3, 1 / 3, 0], "A": [0, 0, 0.5], "H": [1 / 3, 1 / 3, 0.5]},
        "tetragonal": {"G": [0, 0, 0], "X": [0.5, 0, 0], "M": [0.5, 0.5, 0], "Z": [0, 0, 0.5]},
    }
    selected = points_frac.get(lattice_type, points_frac["cubic"])
    return {label: np.array(coord) @ vecs for label, coord in selected.items()}


def _normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Cannot normalize near-zero vector.")
    return v / n


def _hex_bz_wireframe(vecs):
    """
    Build a hexagonal-prism wireframe approximation of first BZ for hex reciprocal lattice.
    """
    b1, b2, b3 = np.asarray(vecs, dtype=float)
    e1 = _normalize(b1)
    ez = _normalize(np.cross(b1, b2))
    e2 = _normalize(np.cross(ez, e1))

    bmag = 0.5 * (np.linalg.norm(b1) + np.linalg.norm(b2))
    r_hex = bmag / np.sqrt(3.0)
    c_half = 0.5 * b3

    thetas = np.deg2rad(np.arange(0, 360, 60))
    base = np.array([r_hex * (np.cos(t) * e1 + np.sin(t) * e2) for t in thetas])

    bottom = base - c_half
    top = base + c_half

    edges = []
    for i in range(6):
        j = (i + 1) % 6
        edges.append((bottom[i], bottom[j]))
        edges.append((top[i], top[j]))
        edges.append((bottom[i], top[i]))
    return edges


def _hex_ws_vertex_mask(kpts, vecs, tol=1e-10):
    """
    Return mask of vertices inside the hexagonal first-BZ prism centered at Gamma.
    """
    b1, b2, b3 = np.asarray(vecs, dtype=float)
    kpts = np.asarray(kpts, dtype=float)

    # In-plane WS constraints from shortest reciprocal vectors.
    g_list = [b1, b2, b1 - b2]
    mask = np.ones(kpts.shape[0], dtype=bool)
    for g in g_list:
        rhs = 0.5 * np.dot(g, g) + tol
        mask &= np.abs(kpts @ g) <= rhs

    # Prism cap constraints along c*.
    c_hat = _normalize(np.cross(b1, b2))
    c_half = 0.5 * abs(np.dot(b3, c_hat)) + tol
    mask &= np.abs(kpts @ c_hat) <= c_half

    return mask


def draw_bz_boundary(ax, lattice_type, vecs):
    """Draw reciprocal-space boundary. Hexagonal draws first-BZ hex prism."""
    if lattice_type == "hexagonal":
        for p0, p1 in _hex_bz_wireframe(vecs):
            ax.plot3D(*zip(p0, p1), color="black", lw=1.8, alpha=0.95)
        return

    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    box_verts = (corners - 0.5) @ vecs
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        ax.plot3D(*zip(box_verts[i], box_verts[j]), color="black", lw=1.2, alpha=0.7)


def plot_fermi_surface(data, band_idx, lattice_type="cubic", fold_ws=True):
    ef = data["fermi_energy"]
    grid = data["band_data"][band_idx]
    nx, ny, nz = data["grid_dimensions"]
    vecs = np.array(data["vectors"], dtype=float)

    verts, faces, _, _ = marching_cubes(grid, ef)

    frac = np.column_stack(
        [
            verts[:, 0] / (nx - 1),
            verts[:, 1] / (ny - 1),
            verts[:, 2] / (nz - 1),
        ]
    )

    origin = np.array(data.get("origin", [0.0, 0.0, 0.0]), dtype=float)
    k_cart = origin + frac @ vecs

    # Keep marching-cubes connectivity in one periodic image, then clip to WS.
    verts_plot = (frac - 0.5) @ vecs
    faces_plot = faces
    if fold_ws and lattice_type == "hexagonal":
        in_ws = _hex_ws_vertex_mask(verts_plot, vecs)
        faces_plot = faces[np.all(in_ws[faces], axis=1)]

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

    draw_bz_boundary(ax, lattice_type, vecs)

    hs_points = get_high_symmetry_points(lattice_type, vecs)
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

    bxsf_data = read_bxsf(path)
    print(f"Loaded {len(bxsf_data['band_data'])} bands.")

    band_input = int(input("Band index to plot: "))
    fold_input = input("Fold surface into first BZ? (Y/n): ").strip().lower()
    fold_ws = fold_input not in ("n", "no")

    plot_fermi_surface(bxsf_data, band_input, lat_type, fold_ws=fold_ws)
