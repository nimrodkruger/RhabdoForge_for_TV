"""
Coordinate conventions between a TrilobiteVision (TV) calibration export and RhabdoForge.

TV head frame (rays.json 'frame'): x along native sensor u, y along native v (image down),
z toward the scene, millimetres. It is the right-handed OpenCV camera convention.

RhabdoForge world/agent frame: +X right, +Y up, -Z forward (OpenGL). Scaffold positions are
micrometres and examples call `model.scale(1e-6)` to reach metres.

The map TV -> RF is the proper rotation M = diag(1, -1, -1) (180 deg about x, det +1) and a unit
change mm -> um (x 1000). Applied to the calibrated viewpoints at full scale, it keeps the model's
effective viewpoint geometry, so the parallax that RF renders matches the calibration's forward
projection. TV viewpoints are effective virtual-camera quantities (they spread about +-75 mm, far
wider than the physical lens), not physical lens positions.

Tangent frames. RF reconstructs each receptor's sampling frame in dispatch.comp as
    T = normalise(right - F (right . F)),   B = F x T.
TV's native basis at a centre direction c is
    e_u = normalise(X - (X . c) c),          e_v = c x e_u          (fieldsample/chart.py)
With M a proper rotation, M (c x e_u) = (M c) x (M e_u), so (T, B, F) = M (e_u, e_v, c) up to a
rotation about F by the angle between RF's `right` (built perpendicular to world up, tangent_frames)
and M e_u (X projected onto the tangent plane). For the left-head axes (within 0.9 deg of -Z) that
angle is at most ~1.1e-4 rad (0.006 deg): RF's first (minor) acceptance axis follows native +u and B
native +v, and gnomonic offsets (tan a, tan b) agree between both charts to that rotation. A replay
that builds TV requests should pass RF's actual T as the request `tangent`, which removes even that.

Tilt. TV rotates the kernel's first axis from e_u toward +e_v by `tilt`. RF rotates the sampled
offset by ioa_tilt with GLSL mat2(c, -s, s, c) (column-major), i.e. from T toward -B. Read from the
code, RF ioa_tilt = -TV tilt; `tv_tilt_from_rf` / `rf_tilt_from_tv` encode that. It is a reading of
the shader, to be confirmed by a rendered test before any tilted field is compared.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

TV_TO_RF = np.diag([1.0, -1.0, -1.0])
MM_TO_UM = 1000.0


def tv_to_rf_vectors(v: np.ndarray) -> np.ndarray:
    """Rotate (..., 3) TV head-frame vectors (directions or tangents) into the RF frame."""
    return np.asarray(v, dtype=np.float64) @ TV_TO_RF.T


def tv_to_rf_points_um(p_mm: np.ndarray) -> np.ndarray:
    """TV head-frame points in mm -> RF frame points in um."""
    return tv_to_rf_vectors(p_mm) * MM_TO_UM


def rf_to_tv_vectors(v: np.ndarray) -> np.ndarray:
    """Inverse of tv_to_rf_vectors (M is its own inverse)."""
    return np.asarray(v, dtype=np.float64) @ TV_TO_RF


def rf_to_tv_points_mm(p_um: np.ndarray) -> np.ndarray:
    return rf_to_tv_vectors(p_um) / MM_TO_UM


def tv_native_basis(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TV's native (e_u, e_v) at unit direction(s) c (..., 3), in the TV frame."""
    c = np.asarray(c, dtype=np.float64)
    c = c / np.linalg.norm(c, axis=-1, keepdims=True)
    x = np.array([1.0, 0.0, 0.0])
    e_u = x - (c @ x)[..., None] * c
    e_u /= np.linalg.norm(e_u, axis=-1, keepdims=True)
    return e_u, np.cross(c, e_u)


def rf_sampling_frame(forward: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T, B) exactly as dispatch.comp builds them from a receptor direction and its ommatidium's right."""
    F = np.asarray(forward, dtype=np.float64)
    F = F / np.linalg.norm(F, axis=-1, keepdims=True)
    r = np.asarray(right, dtype=np.float64)
    T = r - F * np.sum(r * F, axis=-1, keepdims=True)
    T /= np.linalg.norm(T, axis=-1, keepdims=True)
    return T, np.cross(F, T)


def rf_tilt_from_tv(tv_tilt_rad):
    return -np.asarray(tv_tilt_rad, dtype=np.float64)


def tv_tilt_from_rf(rf_ioa_tilt_rad):
    return -np.asarray(rf_ioa_tilt_rad, dtype=np.float64)


def scaffold_from_rays_meta(meta: dict) -> dict[str, np.ndarray]:
    """
    Native RF scaffold arrays from a TV rays.json document (no TrilobiteVision import).

    Rows follow rays.json 'lenslets' order (their 'index'). Returns positions (N, 3) float32 um,
    directions (N, 3) float32 unit, eye_id (N,) float64 zeros (left head = eye 0), and, separately,
    lenslet_ids (N,) str and source_index (N,) int for the registration (never written to the NPZ).
    """
    if meta.get('frame', {}).get('units') != 'mm':
        raise ValueError("rays.json frame units must be 'mm'")
    lenslets = sorted(meta['lenslets'], key=lambda item: int(item['index']))
    if [int(item['index']) for item in lenslets] != list(range(len(lenslets))):
        raise ValueError('rays.json lenslet indices are not 0..N-1')

    vp = np.array([item['viewpoint_mm'] for item in lenslets], dtype=np.float64)
    c = np.array([item['centre_direction'] for item in lenslets], dtype=np.float64)
    if not (np.isfinite(vp).all() and np.isfinite(c).all()):
        raise ValueError('non-finite viewpoint or centre direction in rays.json')
    c /= np.linalg.norm(c, axis=1, keepdims=True)

    return {
        'positions': tv_to_rf_points_um(vp).astype(np.float32),
        'directions': tv_to_rf_vectors(c).astype(np.float32),
        'eye_id': np.zeros(len(lenslets), dtype=np.float64),
        'lenslet_ids': np.array([item['lenslet_id'] for item in lenslets]),
        'source_index': np.array([int(item['index']) for item in lenslets]),
    }


def match_rows(model_positions: np.ndarray, scaffold_positions: np.ndarray, scale: float = 1.0,
               tolerance: float = 1.0) -> np.ndarray:
    """
    Source scaffold row of every post-construction model row, by exact position match.

    Model construction reorders facets (canonical order), so row identity must be recovered.
    Positions are distinct viewpoints, so the nearest scaffold position identifies the row.
    `scale` is the factor applied to the model after construction (e.g. 1e-6). `tolerance` is in
    scaffold units (um): float32 positions of ~75 mm round to ~0.01 um, and viewpoints are ~12 mm
    apart. Raises unless the result is an unambiguous one-to-one permutation within tolerance.
    """
    pm = np.asarray(model_positions, dtype=np.float64).reshape(-1, 3) / scale
    ps = np.asarray(scaffold_positions, dtype=np.float64).reshape(-1, 3)
    if pm.shape != ps.shape:
        raise ValueError(f'model has {len(pm)} rows, scaffold {len(ps)}')
    d = np.linalg.norm(pm[:, None, :] - ps[None, :, :], axis=2)
    rows = np.argmin(d, axis=1)
    err = d[np.arange(len(pm)), rows]
    if not np.array_equal(np.sort(rows), np.arange(len(ps))):
        raise ValueError('position match is not one-to-one')
    if err.max() > tolerance:
        raise ValueError(f'position match error {err.max():.3g} um exceeds {tolerance}')
    # Unambiguous: the runner-up must be much further than the match
    d[np.arange(len(pm)), rows] = np.inf
    if np.min(d) < 100 * max(err.max(), 1e-9):
        raise ValueError('ambiguous position match (near-duplicate viewpoints)')
    return rows


def ids_for_rows(rows: Sequence[int], lenslet_ids: Iterable[str]) -> list[str]:
    ids = list(lenslet_ids)
    return [ids[int(r)] for r in rows]
