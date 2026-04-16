"""Single-plane BXSF slicer based on the current slicer_f workflow.

This keeps the newer interpolation and contour handling from ``slicer_f.py``
but removes the slice sweep. It evaluates one user-specified plane, plots the
Fermi contours on that plane, and prints the area/frequency of every closed
contour.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent
if str(here) not in sys.path:
    sys.path.insert(0, str(here))

from slicer_f import (  # noqa: E402
    _plot_slice_contours,
    area_to_frequency_T,
    contour_segments,
    energies_on_plane_from_bxsf,
    plot_fermi_surface,
    polygon_area,
    read_bxsf,
    ReciprocalLattice,
)


def _normalize(vec: np.ndarray, *, tol: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = np.linalg.norm(arr)
    if norm < tol:
        raise ValueError("Vector has near-zero norm.")
    return arr / norm


def _format_vec(vec: np.ndarray, *, precision: int = 6) -> str:
    return np.array2string(np.asarray(vec, dtype=float), precision=precision, separator=", ")


def slice_one_plane(
    data: dict,
    band_idx: int,
    *,
    center_frac=(0.0, 0.0, 0.0),
    normal_cart=(0.0, 0.0, 1.0),
    offset_along_normal=0.0,
    orient_hint_cart=None,
    shape=(401, 401),
    half_range=(1.5, 1.5),
    area_tol=1e-5,
    show=True,
):
    """Evaluate one plane and report all closed Fermi contours on it."""
    ef = float(data["fermi_energy"])
    lattice = ReciprocalLattice.from_B(np.array(data["vectors"], dtype=float).T)

    center_frac = np.asarray(center_frac, dtype=float).reshape(3)
    n_hat = _normalize(np.asarray(normal_cart, dtype=float))
    base_center_cart = lattice.frac_to_cart(center_frac)
    plane_center_cart = base_center_cart + float(offset_along_normal) * n_hat
    plane_center_frac = lattice.cart_to_frac(plane_center_cart)

    S, T, E2d, u, v, n_plane, lattice = energies_on_plane_from_bxsf(
        data,
        band_idx,
        center_cart=plane_center_cart,
        normal_cart=n_hat,
        orient_hint_cart=orient_hint_cart,
        shape=shape,
        half_range=half_range,
    )

    segments = contour_segments(S, T, E2d, ef)
    closed_records = []
    open_segments = []

    for seg_idx, (vertices, closed) in enumerate(segments, start=1):
        if len(vertices) < 3:
            continue
        if not closed:
            open_segments.append(vertices)
            continue

        area = float(polygon_area(vertices))
        if area <= area_tol:
            continue

        closed_records.append(
            {
                "segment_index": seg_idx,
                "area_Ainv2": area,
                "freq_T": float(area_to_frequency_T(area)),
                "vertices_st": vertices,
            }
        )

    closed_records.sort(key=lambda rec: rec["area_Ainv2"], reverse=True)

    if show:
        title = (
            f"Band {band_idx} single plane: E=E_F contours\n"
            f"offset={float(offset_along_normal):.6g}, closed={len(closed_records)}, open={len(open_segments)}"
        )
        _plot_slice_contours(
            S,
            T,
            E2d,
            ef,
            center_cart=plane_center_cart,
            u=u,
            v=v,
            n_hat=n_plane,
            lattice=lattice,
            title=title,
        )

    print(f"Band index: {band_idx}")
    print(f"Fermi energy: {ef:.8g}")
    print(f"Base center (frac):  {_format_vec(center_frac)}")
    print(f"Plane normal (cart): {_format_vec(n_hat)}")
    print(f"Offset along normal: {float(offset_along_normal):.6g}")
    print(f"Plane center (frac): {_format_vec(plane_center_frac)}")
    print(f"Plane center (cart): {_format_vec(plane_center_cart)}")
    print(f"Total contour segments: {len(segments)}")
    print(f"Closed contours kept: {len(closed_records)}")
    print(f"Open contour segments: {len(open_segments)}")

    if closed_records:
        print("Closed contour areas and dHvA frequencies:")
        for orbit_idx, rec in enumerate(closed_records, start=1):
            print(
                f"  {orbit_idx}. segment={rec['segment_index']}, "
                f"area={rec['area_Ainv2']:.6g} A^-2, "
                f"F={rec['freq_T']:.6g} T ({rec['freq_T'] / 1e3:.6g} kT)"
            )
    else:
        print("No closed E=E_F contours found on this plane.")

    return {
        "band_idx": int(band_idx),
        "fermi_energy": ef,
        "center_frac": center_frac,
        "normal_cart": n_hat,
        "offset_along_normal": float(offset_along_normal),
        "plane_center_frac": plane_center_frac,
        "plane_center_cart": plane_center_cart,
        "S": S,
        "T": T,
        "E2d": E2d,
        "closed_contours": closed_records,
        "open_segments": open_segments,
    }


if __name__ == "__main__":
    path = input("Path to .bxsf file: ").strip()
    data = read_bxsf(path)
    print(f"Loaded {len(data['band_data'])} bands. E_F = {data['fermi_energy']}")

    band = int(input("Band index: ").strip())

    do3d = input("Plot 3D Fermi surface first? (y/N): ").strip().lower() in ("y", "yes")
    if do3d:
        plot_fermi_surface(data, band)

    c_in = input("Base center (fractional b1 b2 b3) [0 0 0]: ").strip()
    center_frac = tuple(map(float, c_in.split())) if c_in else (0.0, 0.0, 0.0)

    n_in = input("Plane normal (Cartesian) [0 0 1]: ").strip()
    normal_cart = tuple(map(float, n_in.split())) if n_in else (0.0, 0.0, 1.0)

    t_in = input("Offset along the normal [0]: ").strip()
    offset_along_normal = float(t_in) if t_in else 0.0

    o_in = input("Orient hint (Cartesian) [Enter for auto]: ").strip()
    orient_hint = tuple(map(float, o_in.split())) if o_in else None

    r_in = input("Half-range rx ry (A^-1) [1.5 1.5]: ").strip()
    half_range = tuple(map(float, r_in.split())) if r_in else (1.5, 1.5)

    s_in = input("Grid Nx Ny [401 401]: ").strip()
    shape = tuple(map(int, s_in.split())) if s_in else (401, 401)

    a_in = input("Area cutoff for closed contours [1e-5]: ").strip()
    area_tol = float(a_in) if a_in else 1e-5

    slice_one_plane(
        data,
        band,
        center_frac=center_frac,
        normal_cart=normal_cart,
        offset_along_normal=offset_along_normal,
        orient_hint_cart=orient_hint,
        half_range=half_range,
        shape=shape,
        area_tol=area_tol,
        show=True,
    )
