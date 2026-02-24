# bzplotter/plane.py
"""
Plane slicing utilities: define and generate 2D sampling grids embedded in 3D k-space.

Math purpose:
- Represent an affine plane patch in R^3 and generate a structured grid of points on it:
    k(s,t) = center + s*û + t*v̂

Physical purpose (quantum oscillations / Fermi-surface cross-sections):
- A dHvA/SdH orbit for field direction B comes from intersecting the Fermi surface
  E(k)=E_F with planes perpendicular to B. Sampling energies on such a plane yields
  a 2D field E(s,t). The contour E(s,t)=E_F gives closed loops whose areas determine
  quantum-oscillation frequencies via the Onsager relation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .linalg import orthonormal_plane_basis

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaneSpec:
    """
    Immutable spec for a rectangular plane patch embedded in 3D k-space.

    Math purpose:
    - Bundle parameters that define a plane and a rectangular sampling domain in
      plane coordinates (s,t).

    Physical purpose:
    - Encodes a reciprocal-space slice used to compute a Fermi-surface cross-section.
      Typical mapping:
        normal ≈ B direction (slice plane ⟂ B)
        center selects region (e.g. around Γ/K)
        half_range + shape control orbit resolution and coverage.
    """
    center: np.ndarray               # (3,) Cartesian k (Å^-1)
    normal: np.ndarray               # (3,) Cartesian direction (will be normalized)
    orient_hint: np.ndarray          # (3,) Cartesian hint for in-plane "x-axis"
    shape: Tuple[int, int]           # (Nx, Ny)
    half_range: Tuple[float, float]  # (rx, ry) extents in plane coords (Å^-1)


def plane_grid(
    spec: PlaneSpec,
    *,
    radial_cutoff: Optional[float] = None,
    return_mesh: bool = True,
    return_basis: bool = False,
) -> tuple:
    """
    Generate a structured grid of Cartesian k-points lying in a specified plane.

    Math purpose:
    - Create a grid over (s,t) ∈ [-rx,rx]×[-ry,ry], then map each (s,t) to:
        k(s,t) = center + s*û + t*v̂
      Optionally apply a disk mask s^2 + t^2 ≤ r_cut^2.

    Physical purpose:
    - Produces the k-points at which you will evaluate band energies E(k) (from
      EIGENVAL or BXSF). The resulting 2D energy map enables extraction of the
      Fermi contour (E=E_F) and its enclosed area (→ quantum oscillation frequency).

    Parameters
    ----------
    spec:
        PlaneSpec defining the plane and rectangular sampling patch.
    radial_cutoff:
        If not None, keep only points inside radius r_cut in plane coords (s,t).
        Useful for focusing on one pocket and reducing boundary effects.
    return_mesh:
        If True, return the 2D coordinate arrays (S,T) so energies can be reshaped
        and contoured in 2D later.
    return_basis:
        If True, return the orthonormal basis vectors (û, v̂, n̂).

    Returns
    -------
    tuple
        If return_mesh is True (default):
            pts : (N,3) Cartesian k-points (flattened; masked if radial_cutoff set)
            S, T: (Ny,Nx) plane-coordinate grids
            mask: (Ny*Nx,) boolean mask (only if radial_cutoff set)
        If return_mesh is False:
            pts : (N,3)

        If return_basis is True, the tuple also ends with:
            u, v, n : (3,) unit vectors defining the plane basis.
    """
    center = np.asarray(spec.center, dtype=float).reshape(3)
    normal = np.asarray(spec.normal, dtype=float).reshape(3)
    orient = np.asarray(spec.orient_hint, dtype=float).reshape(3)

    Nx, Ny = spec.shape
    rx, ry = spec.half_range

    # Build orthonormal in-plane axes û, v̂ and plane normal n̂.
    u, v, n = orthonormal_plane_basis(normal=normal, orient_hint=orient)

    # Plane coordinates (s,t) on a structured grid.
    s = np.linspace(-rx, rx, Nx)
    t = np.linspace(-ry, ry, Ny)
    S, T = np.meshgrid(s, t, indexing="xy")  # (Ny,Nx)

    # Embed the 2D grid into 3D k-space.
    pts_full = center + S[..., None] * u + T[..., None] * v  # (Ny,Nx,3)
    pts_full = pts_full.reshape(-1, 3)                       # (Ny*Nx,3)

    mask = None
    if radial_cutoff is not None:
        r = np.sqrt(S**2 + T**2).reshape(-1)  # radius in plane coords
        mask = r <= radial_cutoff
        pts = pts_full[mask]
    else:
        pts = pts_full

    out = [pts]

    if return_mesh:
        out.extend([S, T])
        if mask is not None:
            out.append(mask)

    if return_basis:
        out.extend([u, v, n])

    return tuple(out)