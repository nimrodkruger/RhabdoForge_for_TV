"""
Physical (hardware) eyes in RhabdoForge: calibration assets and coordinate conventions.

Nothing here imports TrilobiteVision. A hardware eye is an ordinary RF scaffold (assets/*.npz)
built by an ordinary Python model config (examples/trilobite_eye.py); this package only registers
where its calibration came from and how its frames relate.
"""
from .assets import AssetError, HardwareAsset, list_assets, load_asset, store_root
from .frames import (
    TV_TO_RF, match_rows, rf_sampling_frame, rf_tilt_from_tv, scaffold_from_rays_meta,
    tv_native_basis, tv_tilt_from_rf, tv_to_rf_points_um, tv_to_rf_vectors,
)

__all__ = [
    'AssetError', 'HardwareAsset', 'list_assets', 'load_asset', 'store_root',
    'TV_TO_RF', 'match_rows', 'rf_sampling_frame', 'rf_tilt_from_tv', 'scaffold_from_rays_meta',
    'tv_native_basis', 'tv_tilt_from_rf', 'tv_to_rf_points_um', 'tv_to_rf_vectors',
]
