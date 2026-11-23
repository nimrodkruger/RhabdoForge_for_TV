"""
TrilobiteVision left head: native RF scaffold from a registered hardware calibration asset.

One ommatidium per calibrated lenslet viewport (130 for the left-room calibration):
    position  = the lenslet's calibrated viewpoint (TV head frame, mm) -> RF frame, um
    direction = the lenslet's measured centre axis (box-centre ray) -> RF frame
    eye_id    = 0 (left head; the right head is not calibrated)

Frame map (rhabdoforge/hardware/frames.py): RF = diag(1, -1, -1) . TV, mm -> um. Viewpoints are kept at
full scale so rendered parallax matches the calibration's forward projection. They are effective
virtual-camera quantities (+-75 x +-57 mm), not physical lens positions.

These centre axes are nearly parallel (all within ~0.9 deg, neighbours 0.12 deg apart): distinct
viewpoints looking through a narrow common window (+-5.3 deg per lenslet). RF's automatic lattice
estimates (apertures, IOA, acceptance, lattice tilt) are therefore meaningless for this eye; the model
config (examples/trilobite_eye.py) sets them explicitly.

The archive holds only the native keys (positions, directions, eye_id) with the native dtypes. The
lenslet IDs, calibration identity and recording paths stay in the asset manifest
(rhabdoforge/hardware/manifests/), never in the scaffold.

Run from the RhabdoForge root, after `python -m rhabdoforge.hardware import <asset> --from <TV root>`:
    python morphological_scaffolds/trilobite_left/run.py [asset@version]
"""
import sys
from pathlib import Path

import numpy as np

from rhabdoforge.hardware import load_asset, scaffold_from_rays_meta

ASSET = 'trilobite-left@left-room-20261105-scaled004'
SAVE_PATH = Path('assets') / 'trilobite_left_scaffold.npz'


def build_scaffold(asset_ref: str = ASSET) -> dict:
    asset = load_asset(asset_ref)
    asset.require('geometry', hashes=True)
    return scaffold_from_rays_meta(asset.rays_meta())


if __name__ == '__main__':
    ref = sys.argv[1] if len(sys.argv) > 1 else ASSET
    s = build_scaffold(ref)

    np.savez_compressed(SAVE_PATH, positions=s['positions'], directions=s['directions'], eye_id=s['eye_id'])

    d = s['directions'].astype(np.float64)
    spread = np.degrees(np.arccos(np.clip(d @ (d.mean(0) / np.linalg.norm(d.mean(0))), -1, 1)))
    print(f'{ref}: {len(d)} ommatidia -> {SAVE_PATH}')
    print(f'  viewpoint extent (mm): x {np.ptp(s["positions"][:, 0]) / 1e3:.1f}, y {np.ptp(s["positions"][:, 1]) / 1e3:.1f}')
    print(f'  axis spread from mean: max {spread.max():.3f} deg')
