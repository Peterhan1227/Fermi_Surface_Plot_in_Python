# bzplotter/coords.py
"""
Reciprocal-space coordinate transforms (fractional <-> Cartesian).

Math purpose:
- Provide stable linear maps between two coordinate representations:
    - fractional coordinates in a reciprocal-lattice basis (b1,b2,b3)
    - Cartesian coordinates in R^3 (Å^-1)

Physical purpose:
- Most geometry (planes, distances, areas in k-space) is easiest in Cartesian k.
- Many file formats / codes specify k in fractional reciprocal-lattice coords.
  This module lets us cleanly move between representations.

Conventions:
- We represent the reciprocal-basis matrix B as a 3x3 matrix whose COLUMNS are
  the reciprocal lattice vectors (b1,b2,b3) expressed in Cartesian coordinates.
  Then:
      k_cart = B @ k_frac
      k_frac = B^{-1} @ k_cart
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _as_points(x: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Internal helper: coerce input into shape (N,3).

    Math purpose:
    - Normalize input shapes so the core transform can be vectorized.

    Physical purpose:
    - Users may pass a single k-point (3,) or a list/array of k-points (N,3).
      We support both without duplicating logic.

    Returns
    -------
    pts : (N,3) float array
    is_single : bool
        True if original input was a single point (3,), so output should be (3,).
    """
    arr = np.asarray(x, dtype=float)
    if arr.shape == (3,):
        return arr.reshape(1, 3), True
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr, False
    raise ValueError(f"Expected shape (3,) or (N,3); got {arr.shape}.")


@dataclass(frozen=True)
class ReciprocalLattice:
    """
    Linear transform between fractional reciprocal coordinates and Cartesian k.

    Math purpose:
    - Store B and its inverse B^{-1} for fast repeated transforms.

    Physical purpose:
    - In quantum-oscillation slicing we frequently:
        - build planes in Cartesian k-space (Å^-1),
        - convert to/from fractional reciprocal coordinates when needed
          (e.g., interfacing with codes, reading file formats).
    """
    B: np.ndarray     # (3,3): columns are b1,b2,b3 in Cartesian Å^-1
    Binv: np.ndarray  # (3,3): cached inverse

    @classmethod
    def from_B(cls, B: np.ndarray) -> "ReciprocalLattice":
        """
        Construct from reciprocal-basis matrix B (columns = reciprocal vectors).

        Math purpose:
        - Validate matrix shape and compute inverse once.

        Physical purpose:
        - Ensures all later coordinate transforms are consistent and fast.
        """
        B = np.asarray(B, dtype=float)
        if B.shape != (3, 3):
            raise ValueError(f"B must be shape (3,3); got {B.shape}.")
        Binv = np.linalg.inv(B)
        return cls(B=B, Binv=Binv)

    def frac_to_cart(self, k_frac: np.ndarray) -> np.ndarray:
        """
        Convert fractional reciprocal coordinates -> Cartesian k (Å^-1).

        Math purpose:
        - Apply linear map: k_cart = B @ k_frac

        Physical purpose:
        - Cartesian k is the natural space for geometry:
          plane construction, orbit areas, distances, projections.
        """
        pts, is_single = _as_points(k_frac)
        # Vectorized: (N,3) @ (3,3)^T  == (B @ k)^T in batch form
        out = pts @ self.B.T
        return out.reshape(3,) if is_single else out

    def cart_to_frac(self, k_cart: np.ndarray) -> np.ndarray:
        """
        Convert Cartesian k (Å^-1) -> fractional reciprocal coordinates.

        Math purpose:
        - Apply inverse map: k_frac = B^{-1} @ k_cart

        Physical purpose:
        - Fractional reciprocal coords are often how k-grids are indexed/stored
          in electronic-structure outputs and are useful when mapping into a
          regular reciprocal lattice grid (e.g., BXSF).
        """
        pts, is_single = _as_points(k_cart)
        out = pts @ self.Binv.T
        return out.reshape(3,) if is_single else out

    def wrap_frac(self, k_frac: np.ndarray, *, center: float = 0.0) -> np.ndarray:
        """
        Wrap fractional coordinates into a canonical unit cell interval.

        Math purpose:
        - Apply modulo-1 wrapping componentwise:
            if center=0.0 -> wrap into [0,1)
            if center=0.5 -> wrap into [-0.5,0.5)

        Physical purpose:
        - Reciprocal space is periodic. When sampling from gridded data (BXSF),
          wrapping helps map arbitrary k-points back onto the fundamental
          reciprocal cell covered by the grid.
        """
        pts, is_single = _as_points(k_frac)
        if center == 0.0:
            out = pts - np.floor(pts)             # [0,1)
        elif center == 0.5:
            out = (pts + 0.5) - np.floor(pts + 0.5) - 0.5  # [-0.5,0.5)
        else:
            # general center: shift, wrap, shift back
            out = (pts - center) - np.floor(pts - center) + center
        return out.reshape(3,) if is_single else out
