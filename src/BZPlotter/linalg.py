# bzplotter/linalg.py
"""
Small, pure linear-algebra helpers used throughout the project.

Math purpose:
- Provide stable, reusable vector operations (normalization, orthonormal bases).

Physical purpose (context: k-space / Brillouin-zone slicing):
- Build well-defined directions in reciprocal space (e.g., plane normal = B-field
  direction; in-plane axes define the sampling grid for Fermi-surface cross-sections).
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def normalize(v: np.ndarray, *, eps: float = 1e-15) -> np.ndarray:
    """
    Normalize a vector.

    Mathematical purpose:
    - Return v / ||v|| with a safety check against near-zero norm.

    Physical purpose:
    - Ensure direction vectors (e.g., magnetic-field direction, plane normal,
      in-plane axes) are true unit vectors so geometric constructions in k-space
      (areas, distances, projections) have the correct scale and meaning.

    Parameters
    ----------
    v:
        Input vector (typically shape (3,)).
    eps:
        Minimum allowed norm before declaring the vector degenerate.

    Returns
    -------
    np.ndarray
        Unit vector in the same direction as v.

    Raises
    ------
    ValueError
        If ||v|| < eps.
    """
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError(f"Cannot normalize near-zero vector (norm={n:g}).")
    return v / n


def orthonormal_plane_basis(
    normal: np.ndarray,
    orient_hint: np.ndarray,
    *,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct an orthonormal basis (u, v, n) for a plane in 3D.

    Mathematical purpose:
    - Given a plane normal n and an "orientation hint" vector, produce:
        n̂  = normalized normal
        û  = normalized component of orient_hint orthogonal to n̂  (Gram–Schmidt)
        v̂  = n̂ × û  (right-handed completion)
      so that {û, v̂, n̂} is orthonormal.

    Physical purpose:
    - Define a k-space slice plane (typically perpendicular to magnetic field B).
      n̂ represents the slice normal (≈ B direction), while û and v̂ define the
      in-plane coordinate system used to sample energies E(k) and extract the
      Fermi contour E(k)=E_F. The enclosed contour area gives quantum-oscillation
      frequency via the Onsager relation.

    Parameters
    ----------
    normal:
        Plane normal vector (shape (3,)).
    orient_hint:
        Vector indicating desired in-plane "x-direction"; must not be parallel
        to the normal (shape (3,)).
    eps:
        Degeneracy threshold (e.g., orient_hint nearly parallel to normal).

    Returns
    -------
    (u, v, n):
        Orthonormal unit vectors (each shape (3,)).

    Raises
    ------
    ValueError
        If orient_hint is (nearly) parallel to normal or normal is degenerate.
    """
    n = normalize(normal, eps=eps)

    orient_hint = np.asarray(orient_hint, dtype=float)

    # Gram–Schmidt: remove projection of orient_hint onto n.
    # u_raw = orient_hint - (orient_hint·n) n
    u_raw = orient_hint - np.dot(orient_hint, n) * n
    u_norm = np.linalg.norm(u_raw)
    if u_norm < eps:
        raise ValueError(
            "orient_hint is (nearly) parallel to normal; choose a different orient_hint."
        )
    u = u_raw / u_norm

    # Complete right-handed basis: v is in-plane and orthogonal to u.
    v = np.cross(n, u)
    v = normalize(v, eps=eps)

    return u, v, n
