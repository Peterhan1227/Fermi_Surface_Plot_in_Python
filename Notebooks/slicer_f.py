"""BXSF -> Fermi-surface plot + planar cross-section area -> dHvA frequency.

Workflow:
- Read .bxsf (VASPKIT) band energy grids
- (Optional) Plot 3D Fermi surface (marching cubes)
- Define a plane (normal ~ B-field direction)
- Interpolate E(k) from the 3D grid onto a 2D plane grid
- Extract E=E_F contours, compute enclosed areas, convert to dHvA frequencies
"""

from __future__ import annotations

import sys
from itertools import combinations, product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import marching_cubes

repo_src = Path(__file__).resolve().parents[1] / "src"
if str(repo_src) not in sys.path:
    sys.path.insert(0, str(repo_src))

try:
    from BZPlotter.coord import ReciprocalLattice
except Exception:  # pragma: no cover
    try:
        from coord import ReciprocalLattice
    except Exception as e:
        raise ImportError("Could not import ReciprocalLattice (BZPlotter.coord or coord)") from e


try:
    from BZPlotter.plane import PlaneSpec, plane_grid
except Exception:  # pragma: no cover
    try:
        from plane import PlaneSpec, plane_grid
    except Exception as e:
        raise ImportError("Could not import PlaneSpec/plane_grid (BZPlotter.plane or plane)") from e


def read_bxsf(filename: str) -> dict:
    data = {
        "fermi_energy": None,
        "band_data": [],
        "grid_dimensions": None,
        "origin": None,
        "vectors": None,
    }
    bxsf_path = Path(filename)
    if not bxsf_path.exists():
        raise FileNotFoundError(filename)

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


def _grid_period_lengths(grid: np.ndarray, tol=1e-8) -> np.ndarray:
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


def _periodic_supercell_triangles(
    grid: np.ndarray,
    level: float,
    lattice: ReciprocalLattice,
    origin: np.ndarray,
    shift: np.ndarray,
    reps=(3, 3, 3),
):
    """Build periodic isosurface triangles in Cartesian coordinates."""
    periods = _grid_period_lengths(grid).astype(int)
    core = np.asarray(grid)[tuple(slice(0, p) for p in periods)]
    tiled = np.tile(core, reps)
    verts, faces, _, _ = marching_cubes(tiled, level)
    frac = verts / periods[None, :] - (np.asarray(reps, dtype=float) // 2)[None, :]
    verts_cart = lattice.frac_to_cart(origin + frac + shift)
    return verts_cart[faces]


def _lattice_translations(lattice: ReciprocalLattice, nmax=2, include_zero=True):
    nset = np.array(list(product(range(-nmax, nmax + 1), repeat=3)), dtype=float)
    if not include_zero:
        nset = nset[np.any(nset != 0.0, axis=1)]
    return lattice.frac_to_cart(nset)


def _ws_normals(lattice: ReciprocalLattice, nmax=2):
    g = _lattice_translations(lattice, nmax=nmax, include_zero=False)
    g = g[np.linalg.norm(g, axis=1) > 1e-12]
    return np.unique(np.round(g, 12), axis=0)


def fold_to_wigner_seitz(kpts: np.ndarray, lattice: ReciprocalLattice, nmax=2) -> np.ndarray:
    g_vectors = _lattice_translations(lattice, nmax=nmax, include_zero=True)
    disp = kpts[:, None, :] - g_vectors[None, :, :]
    idx = np.argmin(np.sum(disp**2, axis=2), axis=1)
    return kpts - g_vectors[idx]


def _ws_vertex_mask(kpts: np.ndarray, lattice: ReciprocalLattice, nmax=2, tol=1e-10) -> np.ndarray:
    g = _ws_normals(lattice, nmax=nmax)
    rhs = 0.5 * np.sum(g * g, axis=1) + tol
    return np.all(np.abs(kpts @ g.T) <= rhs[None, :], axis=1)


def _compact_polygon_vertices(poly: np.ndarray, tol=1e-10) -> np.ndarray:
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


def _clip_polygon_to_halfspace(
    poly: np.ndarray, normal: np.ndarray, bound: float, tol=1e-10
) -> np.ndarray:
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


def _clip_triangles_to_ws(
    triangles: np.ndarray, lattice: ReciprocalLattice, nmax=2, tol=1e-10
):
    """Clip periodic triangles to the WS cell without creating a seam."""
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


def _wrap_triangles_in_frac_cell(
    k_frac: np.ndarray, faces: np.ndarray, lattice: ReciprocalLattice, center=0.5
):
    tri_frac = k_frac[faces]
    tri_centers = tri_frac.mean(axis=1)
    tri_centers_wrapped = lattice.wrap_frac(tri_centers, center=center)
    shifts = tri_centers - tri_centers_wrapped
    tri_wrapped = tri_frac - shifts[:, None, :]

    verts_wrapped = lattice.frac_to_cart(tri_wrapped.reshape(-1, 3))
    faces_wrapped = np.arange(len(verts_wrapped), dtype=int).reshape(-1, 3)
    return verts_wrapped, faces_wrapped


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


def _project_to_plane_coords(
    pts: np.ndarray, center_cart: np.ndarray, u: np.ndarray, v: np.ndarray
) -> np.ndarray:
    rel = np.asarray(pts, dtype=float) - np.asarray(center_cart, dtype=float)
    return np.column_stack([rel @ u, rel @ v])


def _ws_slice_polygon(
    lattice: ReciprocalLattice,
    center_cart: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    n_hat: np.ndarray,
    nmax=2,
    tol=1e-9,
) -> np.ndarray | None:
    """Return the WS-cell cross-section polygon in the slice (s,t) coordinates."""
    pts = []
    center_cart = np.asarray(center_cart, dtype=float).reshape(3)
    n_hat = np.asarray(n_hat, dtype=float).reshape(3)

    for p0, p1 in _ws_polyhedron_edges(lattice, nmax=nmax):
        d0 = float(np.dot(n_hat, p0 - center_cart))
        d1 = float(np.dot(n_hat, p1 - center_cart))

        if abs(d0) <= tol:
            pts.append(p0)
        if abs(d1) <= tol:
            pts.append(p1)
        if d0 * d1 < -tol * tol:
            t = d0 / (d0 - d1)
            pts.append(p0 + t * (p1 - p0))

    if not pts:
        return None

    pts = np.unique(np.round(np.asarray(pts, dtype=float), 10), axis=0)
    if len(pts) < 3:
        return None

    st = _project_to_plane_coords(pts, center_cart, u, v)
    ctr = st.mean(axis=0)
    ang = np.arctan2(st[:, 1] - ctr[1], st[:, 0] - ctr[0])
    order = np.argsort(ang)
    return st[order]


def plot_fermi_surface(
    data: dict,
    band_idx: int,
    fold_ws: bool = True,
):
    ef = data["fermi_energy"]
    grid = data["band_data"][band_idx]
    lattice = ReciprocalLattice.from_B(np.array(data["vectors"], dtype=float).T)

    origin = np.asarray(data.get("origin", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
    shift = np.zeros(3, dtype=float)

    if fold_ws:
        tri_super = _periodic_supercell_triangles(grid, ef, lattice, origin, shift)
        verts_plot, faces_plot = _clip_triangles_to_ws(tri_super, lattice, nmax=2)
    else:
        periods = _grid_period_lengths(grid)
        verts, faces, _, _ = marching_cubes(grid, ef)
        frac_grid = verts / periods[None, :]
        k_frac = origin + frac_grid
        verts_plot, faces_plot = _wrap_triangles_in_frac_cell(
            k_frac, faces, lattice, center=0.5
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
    ax.set_title(f"Band {band_idx} Fermi Surface")
    ax.set_axis_off()
    xmin, xmax = ax.get_xlim3d()
    ymin, ymax = ax.get_ylim3d()
    zmin, zmax = ax.get_zlim3d()
    ax.set_box_aspect((xmax - xmin, ymax - ymin, zmax - zmin))
    ax.set_proj_type("ortho")
    plt.show()


def interp_trilinear_periodic(grid: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Trilinear interpolation for points in index coordinates (x,y,z)."""
    px, py, pz = _grid_period_lengths(grid).astype(int)

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


def _default_orient_hint_from_normal(normal_cart: np.ndarray) -> np.ndarray:
    n = np.asarray(normal_cart, dtype=float).reshape(3)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-12:
        raise ValueError("normal_cart is near zero; cannot define orient_hint.")
    n = n / n_norm

    ex = np.array([1.0, 0.0, 0.0], dtype=float)
    hint = np.cross(n, ex)
    if np.linalg.norm(hint) < 1e-12:
        ey = np.array([0.0, 1.0, 0.0], dtype=float)
        hint = np.cross(n, ey)
    return hint


def energies_on_plane_from_bxsf(
    data: dict,
    band_idx: int,
    center_cart: np.ndarray,
    normal_cart: np.ndarray,
    orient_hint_cart: np.ndarray | None = None,
    *,
    shape=(401, 401),
    half_range=(1.0, 1.0),
):
    grid = data["band_data"][band_idx]
    lattice = ReciprocalLattice.from_B(np.array(data["vectors"], dtype=float).T)
    periods = _grid_period_lengths(grid)
    normal_arr = np.asarray(normal_cart, dtype=float).reshape(3)
    n_norm = np.linalg.norm(normal_arr)
    if n_norm < 1e-12:
        raise ValueError("normal_cart is near zero; cannot define slice plane.")

    if orient_hint_cart is None:
        orient_arr = _default_orient_hint_from_normal(normal_arr)
    else:
        orient_arr = np.asarray(orient_hint_cart, dtype=float).reshape(3)
        o_norm = np.linalg.norm(orient_arr)
        if o_norm < 1e-12:
            orient_arr = _default_orient_hint_from_normal(normal_arr)
        elif abs(np.dot(normal_arr / n_norm, orient_arr / o_norm)) > 0.999:
            orient_arr = _default_orient_hint_from_normal(normal_arr)

    spec = PlaneSpec(
        center=np.asarray(center_cart, dtype=float).reshape(3),
        normal=normal_arr,
        orient_hint=orient_arr,
        shape=tuple(shape),
        half_range=tuple(half_range),
    )

    pts, S, T, u, v, n = plane_grid(spec, return_mesh=True, return_basis=True)

    origin = np.asarray(data.get("origin", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
    frac = lattice.cart_to_frac(pts) - origin
    frac = frac - np.floor(frac)

    xyz = np.column_stack([
        frac[:, 0] * periods[0],
        frac[:, 1] * periods[1],
        frac[:, 2] * periods[2],
    ])

    e = interp_trilinear_periodic(grid, xyz)
    E2d = e.reshape(S.shape)
    return S, T, E2d, u, v, n, lattice


def polygon_area(xy: np.ndarray) -> float:
    if len(xy) < 3:
        return 0.0
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def contour_segments(
    S: np.ndarray, T: np.ndarray, E2d: np.ndarray, level: float, close_tol=1e-10
):
    fig, ax = plt.subplots()
    cs = ax.contour(S, T, E2d, levels=[level])
    segments = []

    if hasattr(cs, "allsegs") and cs.allsegs:
        for seg in cs.allsegs[0]:
            v = np.asarray(seg)
            if len(v) < 3:
                continue
            closed = np.linalg.norm(v[0] - v[-1]) <= close_tol
            if closed:
                v = v[:-1]
            segments.append((v, closed))
    else:
        for coll in getattr(cs, "collections", []):
            for p in coll.get_paths():
                v = np.asarray(p.vertices)
                if len(v) < 3:
                    continue
                closed = np.linalg.norm(v[0] - v[-1]) <= close_tol
                if closed:
                    v = v[:-1]
                segments.append((v, closed))

    plt.close(fig)
    return segments


def contour_areas(S: np.ndarray, T: np.ndarray, E2d: np.ndarray, level: float):
    areas = []
    for v, closed in contour_segments(S, T, E2d, level):
        if not closed or len(v) < 3:
            continue
        areas.append(polygon_area(v))
    return areas


def _plot_slice_contours(
    S: np.ndarray,
    T: np.ndarray,
    E2d: np.ndarray,
    level: float,
    *,
    center_cart: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    n_hat: np.ndarray,
    lattice: ReciprocalLattice,
    title: str,
):
    segments = contour_segments(S, T, E2d, level)
    ws_poly = _ws_slice_polygon(lattice, center_cart, u, v, n_hat, nmax=2)
    n_closed = sum(int(closed) for _, closed in segments)
    n_open = len(segments) - n_closed

    plt.figure(figsize=(7, 6))
    for seg, closed in segments:
        if len(seg) < 2:
            continue
        plt.plot(
            seg[:, 0],
            seg[:, 1],
            color=("C0" if closed else "C1"),
            ls=("-" if closed else "--"),
            lw=1.5,
        )

    if ws_poly is not None:
        poly = np.vstack([ws_poly, ws_poly[0]])
        plt.plot(poly[:, 0], poly[:, 1], color="black", lw=1.2)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"{title}\nclosed={n_closed}, open={n_open}")
    plt.xlabel("s (A^-1)")
    plt.ylabel("t (A^-1)")
    plt.tight_layout()
    plt.show()

    return segments, n_closed, n_open


def area_to_frequency_T(area_Ainv2: float) -> float:
    hbar = 1.054_571_817e-34
    e = 1.602_176_634e-19
    area_m2inv = area_Ainv2 * 1e20
    return (hbar / (2 * np.pi * e)) * area_m2inv


def slice_and_dhva(
    data: dict,
    band_idx: int,
    *,
    center_frac=(0.0, 0.0, 0.0),
    normal_cart=(0.0, 0.0, 1.0),
    orient_hint_cart=None,
    shape=(401, 401),
    half_range=(1.0, 1.0),
    show=True,
):
    ef = float(data["fermi_energy"])
    normal_arr = np.asarray(normal_cart, dtype=float)
    lattice = ReciprocalLattice.from_B(np.array(data["vectors"], dtype=float).T)
    center_cart = lattice.frac_to_cart(np.asarray(center_frac, dtype=float))

    S, T, E2d, u, v, n_hat, _ = energies_on_plane_from_bxsf(
        data,
        band_idx,
        center_cart=center_cart,
        normal_cart=normal_arr,
        orient_hint_cart=orient_hint_cart,
        shape=shape,
        half_range=half_range,
    )

    areas = contour_areas(S, T, E2d, ef)
    areas = sorted([a for a in areas if a > 0], reverse=True)
    freqs = [area_to_frequency_T(a) for a in areas]

    if show:
        _, n_closed, n_open = _plot_slice_contours(
            S,
            T,
            E2d,
            ef,
            center_cart=center_cart,
            u=u,
            v=v,
            n_hat=n_hat,
            lattice=lattice,
            title=f"Band {band_idx} slice: E=E_F contours\nnormal ~ {normal_arr}",
        )

        if areas:
            a0 = areas[0]
            f0 = freqs[0]
            print(f"Closed contours found: {n_closed}")
            print(f"Open contour fragments: {n_open}")
            print(f"Largest area: {a0:.6g} A^-2")
            print(f"Frequency:    {f0:.6g} T  ({f0/1e3:.6g} kT)")
        else:
            print(f"No closed E=E_F contours found on this slice. Open fragments: {n_open}")

    return areas, freqs


def sweep_slices_and_dhva_max(
    data: dict,
    band_idx: int,
    *,
    center_frac=(0.0, 0.0, 0.0),
    normal_cart=(0.0, 0.0, 1.0),
    orient_hint_cart=None,
    shape=(401, 401),
    half_range=(1.0, 1.0),
    t_min=-1.0,
    t_max=1.0,
    n_slices=101,
    force_include_t0=True,
    show_best_contour=True,
    show_A_vs_t=True,
):
    ef = float(data["fermi_energy"])
    lattice = ReciprocalLattice.from_B(np.array(data["vectors"], dtype=float).T)

    center_cart0 = lattice.frac_to_cart(np.asarray(center_frac, dtype=float))

    n_hat = np.asarray(normal_cart, dtype=float).reshape(3)
    n_norm = np.linalg.norm(n_hat)
    if n_norm < 1e-14:
        raise ValueError("normal_cart has near-zero norm.")
    n_hat = n_hat / n_norm

    n_slices = int(n_slices)
    if n_slices < 1:
        raise ValueError("n_slices must be >= 1.")

    t_min = float(t_min)
    t_max = float(t_max)
    t_vals = np.linspace(t_min, t_max, n_slices)
    if force_include_t0 and (t_min <= 0.0 <= t_max):
        if not np.any(np.isclose(t_vals, 0.0, atol=1e-12, rtol=0.0)):
            t_vals = np.sort(np.unique(np.concatenate([t_vals, np.array([0.0])])))

    best = {
        "t0": None,
        "area_Ainv2": 0.0,
        "freq_T": 0.0,
        "n_contours": 0,
        "center_cart": None,
        "center_frac": None,
        "areas_Ainv2": [],
        "freqs_T": [],
    }
    center_plane = {
        "evaluated": False,
        "area_Ainv2": 0.0,
        "freq_T": 0.0,
        "n_contours": 0,
    }
    A_of_t = []

    for t0 in t_vals:
        center_cart = center_cart0 + t0 * n_hat
        S, T, E2d, *_ = energies_on_plane_from_bxsf(
            data,
            band_idx,
            center_cart=center_cart,
            normal_cart=n_hat,
            orient_hint_cart=orient_hint_cart,
            shape=shape,
            half_range=half_range,
        )

        areas = sorted((a for a in contour_areas(S, T, E2d, ef) if a > 0.0), reverse=True)
        amax = float(areas[0]) if areas else 0.0
        A_of_t.append((float(t0), amax, float(len(areas))))

        if np.isclose(t0, 0.0, atol=1e-12, rtol=0.0):
            center_plane["evaluated"] = True
            center_plane["area_Ainv2"] = amax
            center_plane["freq_T"] = float(area_to_frequency_T(amax)) if amax > 0.0 else 0.0
            center_plane["n_contours"] = int(len(areas))

        if areas and amax > best["area_Ainv2"]:
            best["t0"] = float(t0)
            best["area_Ainv2"] = amax
            best["freq_T"] = float(area_to_frequency_T(amax))
            best["n_contours"] = int(len(areas))
            best["center_cart"] = center_cart
            best["center_frac"] = lattice.cart_to_frac(center_cart)
            best["areas_Ainv2"] = areas
            best["freqs_T"] = [area_to_frequency_T(a) for a in areas]

    A_of_t = np.asarray(A_of_t, dtype=float)

    if show_A_vs_t:
        plt.figure(figsize=(7, 4))
        plt.plot(A_of_t[:, 0], A_of_t[:, 1])
        plt.xlabel("t0 (offset along normal; k-units)")
        plt.ylabel("max area on slice (A^-2 if k is A^-1)")
        plt.title(f"Band {band_idx}: max slice area vs offset")
        plt.tight_layout()
        plt.show()

    if show_best_contour and best["t0"] is not None:
        center_cart_best = best["center_cart"]
        S, T, E2d, u, v, n_best, _ = energies_on_plane_from_bxsf(
            data,
            band_idx,
            center_cart=center_cart_best,
            normal_cart=n_hat,
            orient_hint_cart=orient_hint_cart,
            shape=shape,
            half_range=half_range,
        )

        _, n_closed, n_open = _plot_slice_contours(
            S,
            T,
            E2d,
            ef,
            center_cart=center_cart_best,
            u=u,
            v=v,
            n_hat=n_best,
            lattice=lattice,
            title=(
                f"Best slice (t0={best['t0']:.6g}): E=E_F contours\n"
                f"Amax={best['area_Ainv2']:.6g} A^-2, F={best['freq_T']:.6g} T"
            ),
        )

    if best["t0"] is None:
        print("No closed E=E_F contours found on any slice in the sweep.")
    else:
        frac_str = np.array2string(best["center_frac"], precision=6, separator=", ")
        cart_str = np.array2string(best["center_cart"], precision=6, separator=", ")
        print(f"Sweep complete: {len(t_vals)} slices")
        print(f"Best offset t0: {best['t0']:.6g}")
        print(f"Best center (frac): {frac_str}")
        print(f"Best center (cart): {cart_str}")
        print(f"Max area: {best['area_Ainv2']:.6g} A^-2")
        print(f"Frequency: {best['freq_T']:.6g} T ({best['freq_T']/1e3:.6g} kT)")
        print(f"Contours on best slice: {best['n_contours']}")
        if center_plane["evaluated"]:
            print(
                f"Center plane (t0=0): {center_plane['area_Ainv2']:.6g} A^-2, "
                f"{center_plane['freq_T']:.6g} T, "
                f"contours={center_plane['n_contours']}"
            )
        print(
            "All contour areas on best slice (A^-2): "
            + ", ".join(f"{a:.6g}" for a in best["areas_Ainv2"])
        )
        print(
            "All dHvA frequencies on best slice (T): "
            + ", ".join(f"{f:.6g}" for f in best["freqs_T"])
        )

    return best, A_of_t


if __name__ == "__main__":
    path = input("Path to .bxsf file: ").strip()
    data = read_bxsf(path)
    print(f"Loaded {len(data['band_data'])} bands. E_F = {data['fermi_energy']}")

    band = int(input("Band index: ").strip())

    do3d = input("Plot 3D Fermi surface first? (y/N): ").strip().lower() in ("y", "yes")
    if do3d:
        plot_fermi_surface(data, band)

    c_in = input("Center (fractional b1 b2 b3) [0 0 0]: ").strip()
    center_frac = tuple(map(float, c_in.split())) if c_in else (0.0, 0.0, 0.0)

    n_in = input("Normal (Cartesian) [0 0 1]: ").strip()
    normal_cart = tuple(map(float, n_in.split())) if n_in else (0.0, 0.0, 1.0)
    o_in = input("Orient hint (Cartesian) [Enter for auto: cross(normal, [1 0 0])]: ").strip()
    orient_hint = tuple(map(float, o_in.split())) if o_in else None

    r_in = input("Half-range rx ry (A^-1) [1.5 1.5]: ").strip()
    half_range = tuple(map(float, r_in.split())) if r_in else (1.5, 1.5)

    s_in = input("Grid Nx Ny [401 401]: ").strip()
    shape = tuple(map(int, s_in.split())) if s_in else (401, 401)

    sweep_in = input("Sweep t_min t_max n_slices [-1 1 101]: ").strip()
    if sweep_in:
        t_min, t_max, n_slices = sweep_in.split()
        t_min, t_max, n_slices = float(t_min), float(t_max), int(n_slices)
    else:
        t_min, t_max, n_slices = -1.0, 1.0, 101

    sweep_slices_and_dhva_max(
        data,
        band,
        center_frac=center_frac,
        normal_cart=normal_cart,
        orient_hint_cart=orient_hint,
        half_range=half_range,
        shape=shape,
        t_min=t_min,
        t_max=t_max,
        n_slices=n_slices,
        show_best_contour=True,
        show_A_vs_t=True,
    )
