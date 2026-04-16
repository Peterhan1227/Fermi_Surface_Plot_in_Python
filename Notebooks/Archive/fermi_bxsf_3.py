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
    bxsf_path = Path(filename)
    if not bxsf_path.exists():
        raise FileNotFoundError(f"File {filename} not found.")

    with bxsf_path.open("r", encoding="utf-8") as f:
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


def _grid_period_lengths(grid, tol=1e-8):
    """
    Return the true periodic sample counts along each axis.

    Some BXSF grids repeat the first plane at the last index (period N-1), while
    others store N unique samples on [0,1). We detect that from the endpoint data.
    """
    arr = np.asarray(grid, dtype=float)
    scale = max(float(np.max(np.abs(arr))), 1.0)
    periods = []
    for axis, n in enumerate(arr.shape):
        first = np.take(arr, 0, axis=axis)
        last = np.take(arr, n - 1, axis=axis)
        repeated = np.max(np.abs(first - last)) <= tol * scale
        periods.append(n - 1 if repeated else n)
    return np.asarray(periods, dtype=float)


def _periodic_supercell_triangles(grid, level, lattice, origin, shift, reps=(3, 3, 3)):
    """Build periodic isosurface triangles in Cartesian coordinates."""
    periods = _grid_period_lengths(grid).astype(int)
    core = np.asarray(grid)[tuple(slice(0, p) for p in periods)]
    tiled = np.tile(core, reps)
    verts, faces, _, _ = marching_cubes(tiled, level)
    frac = verts / periods[None, :] - (np.asarray(reps, dtype=float) // 2)[None, :]
    verts_cart = lattice.frac_to_cart(origin + frac + shift)
    return verts_cart[faces]


def _lattice_translations(lattice, nmax=3, include_zero=True):
    """Build reciprocal vectors n1*b1+n2*b2+n3*b3 in Cartesian coordinates."""
    nset = np.array(list(product(range(-nmax, nmax + 1), repeat=3)), dtype=float)
    if not include_zero:
        nset = nset[np.any(nset != 0.0, axis=1)]
    return lattice.frac_to_cart(nset)


def _ws_normals(lattice, nmax=3):
    """Return unique non-zero WS plane normals from reciprocal translations."""
    g = _lattice_translations(lattice, nmax=nmax, include_zero=False)
    g = g[np.linalg.norm(g, axis=1) > 1e-12]
    return np.unique(np.round(g, 12), axis=0)


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


def _compact_polygon_vertices(poly, tol=1e-10):
    """Drop duplicate consecutive vertices introduced during clipping."""
    if len(poly) == 0:
        return poly

    out = [poly[0]]
    for pt in poly[1:]:
        if np.linalg.norm(pt - out[-1]) > tol:
            out.append(pt)

    if len(out) > 1 and np.linalg.norm(out[0] - out[-1]) <= tol:
        out.pop()
    return np.asarray(out, dtype=float)


def _clip_polygon_to_halfspace(poly, normal, bound, tol=1e-10):
    """Clip a polygon against one WS half-space normal.x <= bound."""
    if len(poly) == 0:
        return poly

    out = []
    prev = poly[-1]
    prev_val = np.dot(normal, prev) - bound
    prev_in = prev_val <= tol

    for curr in poly:
        curr_val = np.dot(normal, curr) - bound
        curr_in = curr_val <= tol

        if curr_in != prev_in:
            denom = prev_val - curr_val
            if abs(denom) > tol:
                t = prev_val / denom
                out.append(prev + t * (curr - prev))

        if curr_in:
            out.append(curr)

        prev = curr
        prev_val = curr_val
        prev_in = curr_in

    if not out:
        return np.empty((0, 3), dtype=float)
    return _compact_polygon_vertices(np.asarray(out, dtype=float), tol=tol)


def _clip_triangles_to_ws(triangles, lattice, nmax=2, tol=1e-10):
    """
    Clip folded triangles to the WS cell instead of dropping straddling faces.

    Dropping any triangle with one vertex outside creates artificial cracks on
    coarse meshes; clipping preserves a continuous boundary-terminated surface.
    """
    normals = _ws_normals(lattice, nmax=nmax)
    bvals = 0.5 * np.sum(normals * normals, axis=1)

    clipped = []
    for tri in triangles:
        poly = np.asarray(tri, dtype=float)
        for normal, bound in zip(normals, bvals):
            poly = _clip_polygon_to_halfspace(poly, normal, bound, tol=tol)
            if len(poly) < 3:
                break

        if len(poly) < 3:
            continue

        anchor = poly[0]
        for i in range(1, len(poly) - 1):
            clipped.append(np.vstack([anchor, poly[i], poly[i + 1]]))

    if not clipped:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)

    verts = np.asarray(clipped, dtype=float).reshape(-1, 3)
    faces = np.arange(len(verts), dtype=int).reshape(-1, 3)
    return verts, faces


def _ws_polyhedron_edges(lattice, nmax=3, tol=1e-9):
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


def draw_bz_boundary(ax, lattice, nmax=3):
    """Draw first Brillouin-zone boundary from WS construction."""
    for p0, p1 in _ws_polyhedron_edges(lattice, nmax=nmax):
        ax.plot3D(*zip(p0, p1), color="black", lw=1.3, alpha=0.8)


def _wrap_triangles_in_frac_cell(k_frac, faces, lattice, center=0.5):
    """
    Wrap triangles coherently into one periodic image to avoid seam-spanning faces.

    We choose one lattice translation per triangle from its center, then apply that
    translation to all three triangle vertices.
    """
    tri_frac = k_frac[faces]  # (Nf,3,3)
    tri_centers = tri_frac.mean(axis=1)
    tri_centers_wrapped = lattice.wrap_frac(tri_centers, center=center)
    shifts = tri_centers - tri_centers_wrapped
    tri_wrapped = tri_frac - shifts[:, None, :]

    verts_wrapped = lattice.frac_to_cart(tri_wrapped.reshape(-1, 3))
    faces_wrapped = np.arange(len(verts_wrapped), dtype=int).reshape(-1, 3)
    return verts_wrapped, faces_wrapped


def plot_fermi_surface(
    data,
    band_idx,
    lattice_type="cubic",
    fold_ws=True,
    points_file=None,
    *,
    wrap_center=0.5,
    manual_shift=(0.0, 0.0, 0.0),
):
    ef = data["fermi_energy"]
    grid = data["band_data"][band_idx]
    vecs = np.array(data["vectors"], dtype=float)
    lattice = ReciprocalLattice.from_B(vecs.T)

    origin = np.asarray(data.get("origin", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
    shift = np.asarray(manual_shift, dtype=float).reshape(3)

    if fold_ws:
        tri_super = _periodic_supercell_triangles(grid, ef, lattice, origin, shift)
        verts_plot, faces_plot = _clip_triangles_to_ws(tri_super, lattice, nmax=2)
    else:
        periods = _grid_period_lengths(grid)
        verts, faces, _, _ = marching_cubes(grid, ef)
        frac_grid = verts / periods[None, :]
        k_frac = origin + frac_grid + shift

        if wrap_center is None:
            verts_plot = lattice.frac_to_cart(k_frac)
            faces_plot = faces
        else:
            verts_plot, faces_plot = _wrap_triangles_in_frac_cell(
                k_frac, faces, lattice, center=wrap_center
            )

    if len(faces_plot) == 0:
        raise ValueError("No Fermi-surface triangles remain after WS clipping.")

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

    wrap_label = {None: "none", 0.5: "gamma", 0.0: "corner"}.get(wrap_center, f"{wrap_center:g}")

    ax.set_title(
        f"Band {band_idx} Fermi Surface\n"
        f"Lattice: {lattice_type.capitalize()} | Fold to WS: {fold_ws} | Wrap: {wrap_label}"
    )
    ax.set_axis_off()
    xmin, xmax = ax.get_xlim3d()
    ymin, ymax = ax.get_ylim3d()
    zmin, zmax = ax.get_zlim3d()
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
    ax.set_proj_type("ortho")  # optional, better for geometry comparison
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

    band_input = int(input("Band index to plot (0-based): "))
    fold_input = input("Fold surface into WS 1st BZ? (Y/n): ").strip().lower()
    fold_ws = fold_input not in ("n", "no")

    wrap_in = input("Wrap for display (gamma/corner/none) [gamma]: ").strip().lower()
    if wrap_in in ("", "g", "gamma", "gamma-centered"):
        wrap_center = 0.5
    elif wrap_in in ("c", "corner", "0"):
        wrap_center = 0.0
    elif wrap_in in ("n", "no", "none", "off"):
        wrap_center = None
    else:
        print("Unknown wrap option; using gamma.")
        wrap_center = 0.5

    shift_in = input("Manual fractional shift in (b1,b2,b3) units [0 0 0]: ").strip()
    if shift_in:
        try:
            manual_shift = tuple(map(float, shift_in.split()))
            if len(manual_shift) != 3:
                raise ValueError
        except ValueError:
            print("Invalid shift; using 0 0 0.")
            manual_shift = (0.0, 0.0, 0.0)
    else:
        manual_shift = (0.0, 0.0, 0.0)

    plot_fermi_surface(
        bxsf_data,
        band_input,
        lat_type,
        fold_ws=fold_ws,
        points_file=points_file,
        wrap_center=wrap_center,
        manual_shift=manual_shift,
    )
