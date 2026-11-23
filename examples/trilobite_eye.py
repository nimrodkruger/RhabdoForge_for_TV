"""
TrilobiteVision left head as an ordinary RhabdoForge eye (native model config, no TrilobiteVision import).

    python examples/trilobite_eye.py                 # headless, prints a short output summary
    python examples/trilobite_eye.py --interactive   # viewport + dashboard

build_model() follows the examples/dynamics_POC.py convention: an executable Python preset that loads
a native scaffold with Model.from_file and returns the built model in metres.

What is calibrated, what is chosen
----------------------------------
Calibrated (asset trilobite-left@left-room-20261105-scaled004, provisional, via the scaffold):
    one ommatidium per lenslet viewport: viewpoint position and measured centre axis.
Chosen here (synthetic, disclosed, adjustable):
    - Acceptance: explicit Gaussian, FWHM_DEG (default 1 deg circular). RF's curr_acc_angles are full
      widths at half maximum (shader: exp(-4 ln2 theta^2 / acc^2)), the same convention as TV's
      GaussianKernel. At 1 deg the left head has ~7.7 px per FWHM.
    - Acceptance axes: lattice tilt forced to 0, so the minor (first) axis follows native +u and the
      second follows native +v (see rhabdoforge/hardware/frames.py).
    - Lens aperture APERTURE_UM and focal length FOCAL_UM: insect-scale physiology (F ~1.29). They set
      the lever arm that turns receptor tip offsets and microsaccades (um) into angles; they are not
      the camera's optics. The camera's viewpoints are effective model quantities in mm.
    - IOA: the measured median angle between neighbouring centre axes (0.12 deg). Passed explicitly
      because RF's estimate from this near-parallel lattice is degenerate. It feeds no acceptance or
      aperture here (both explicit).
    - Microsaccade lateral amplitude scaled so full drive moves a receptor by ~MAX_SACCADE_DEG,
      inside the lenslet's +-5.3 deg window.
    - Bundles:
        'single' (baseline): one achromatic receptor per lenslet on the centre axis.
        'open7'  (intended): seven achromatic receptor slots per lenslet, open (not fused), placed
                 like Drosophila R1-R7 but scaled so the outer ring sits ~OPEN7_RING_DEG off axis.
                 No neural superposition: neighbouring lenslets look the same way, so there is no
                 cross-lenslet wiring to make. Bundle orientation is explicit (chi = 0, chirality +1).
    - Sensitivity 1 on R, G, B for every receptor (achromatic; the camera is monochrome).

Auto-derived apertures (~11 mm from the 12 mm viewpoint spacing), the acceptance that follows from
them (0.009 to 5.7 deg depending on the bundle) and the lattice tilt (-28 to +29 deg) are meaningless
for this eye; tests/hardware checks that this preset never uses them.

The centre axes are the rays through each lenslet's ROI box centre (TV's viewport reference
direction). Their ~0.9 deg spread is an exact, isotropic linear function of viewpoint rotated by
90 deg, consistent with the axis-aligned box grid sitting on the fitted lattice rotated by -0.0089 rad:
the spread reflects the box convention more than distinct optical axes.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from rhabdoforge.compound_eyes import Model
from rhabdoforge.compound_eyes.helpers.acceptance import ExplicitAcceptance
from rhabdoforge.compound_eyes.rhabdomeres import RhabdomereBundle

SCAFFOLD = 'assets/trilobite_left_scaffold.npz'
ASSET = 'trilobite-left@left-room-20261105-scaled004'   # calibration the scaffold was built from

FWHM_DEG = 1.0
APERTURE_UM = 16.0
FOCAL_UM = 20.6
MAX_SACCADE_DEG = 1.0
OPEN7_RING_DEG = 1.5
BUNDLES = ('single', 'open7')

# Drosophila R1-R7 tip offsets (um, rhabdomeres.drosophila_bundle), used only for the open7 layout
_DROSO_OFFSETS = np.array([
    [-1.6881, 1.0273], [-1.8046, -0.9934], [-1.7111, -2.9717], [-0.0025, -1.9261],
    [1.6690, -0.9493], [1.6567, 0.9762], [0.0045, -0.0113]])


def make_bundle(kind: str = 'single') -> RhabdomereBundle:
    lateral_um = FOCAL_UM * np.tan(np.deg2rad(MAX_SACCADE_DEG))
    common = dict(focal_um=FOCAL_UM, ref_aperture_um=APERTURE_UM, sensitivity=1.0, wavelengths_nm=540.0,
                  ampl_lat_um=lateral_um, fused_rhabdoms=False, superposed_indices=None)
    if kind == 'single':
        return RhabdomereBundle(name='Trilobite single (achromatic)', offsets_um=((0.0, 0.0),), **common)
    if kind == 'open7':
        ring = np.linalg.norm(_DROSO_OFFSETS[:6], axis=1).mean()
        k = FOCAL_UM * np.tan(np.deg2rad(OPEN7_RING_DEG)) / ring
        return RhabdomereBundle(name='Trilobite open7 (achromatic)', offsets_um=_DROSO_OFFSETS * k,
                                diameters_um=2.0 * k, center_index=6, **common)
    raise ValueError(f'bundle must be one of {BUNDLES}')


def measured_ioa_rad(directions: np.ndarray) -> float:
    d = np.asarray(directions, dtype=np.float64)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    c = np.clip(d @ d.T, -1.0, 1.0)
    np.fill_diagonal(c, -1.0)
    return float(np.median(np.arccos(c.max(axis=1))))


def build_model(bundle: str = 'single', fwhm_deg=FWHM_DEG, scaffold: str = SCAFFOLD) -> Model:
    """The hardware eye in metres. fwhm_deg: scalar (circular) or (minor along u, major along v)."""

    with np.load(Path(scaffold) if Path(scaffold).exists() else _root() / scaffold) as s:
        n = len(s['positions'])
        ioa = measured_ioa_rad(s['directions'])

    b = make_bundle(bundle)
    fw = np.broadcast_to(np.deg2rad(np.asarray(fwhm_deg, dtype=np.float64)), (2,))
    acceptance = ExplicitAcceptance(np.broadcast_to(fw, (n, b.count, 2)).copy())

    model = Model.from_file(
        scaffold,
        bundle=b,
        aperture_um=APERTURE_UM,
        interommatidial_angles_rad=[ioa, ioa],
        acceptance=acceptance,
        bundle_orientations=np.zeros(n),
        chiralities=np.ones(n),
    )
    model.interommatidial_tilt = 0.0      # acceptance axes along native u / v, not RF's lattice estimate
    model.scale(1e-6)                     # model constructs in micrometres
    return model


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', choices=BUNDLES, default='single')
    ap.add_argument('--fwhm-deg', type=float, default=FWHM_DEG)
    ap.add_argument('--interactive', action='store_true')
    ap.add_argument('--steps', type=int, default=30)
    a = ap.parse_args(argv)

    from rhabdoforge.engine import Context, Agent, Scene, Asset
    from rhabdoforge.engine.meshes import CUBE_VERTICES, CUBE_INDICES
    from rhabdoforge.renderers import Renderer

    context = Context()
    scene = Scene(background_color=(0.15, 0.15, 0.3))
    scene.add_sky('assets/textures/kloppenheim_05_4k.exr')
    pos, uvs = np.split(CUBE_VERTICES.reshape(-1, 5), [3], axis=1)
    crate = Asset.from_arrays(name='crate', vertices=pos, faces=CUBE_INDICES.reshape(-1, 3), uv_coords=uvs,
                              texture='assets/textures/wood.jpg', sRGB=False)
    scene.add_instance(asset=crate, transform=(0.0, 0.0, -1.2))        # ~1.2 m ahead, inside the window

    model = build_model(a.bundle, a.fwhm_deg)
    agent = Agent(position=(0.0, 0.0, 0.0))
    renderer = Renderer(model=model, scene=scene, agent=agent, nb_samples=64, time_dithering=False,
                        enable_microsaccades=False)

    if a.interactive:
        while context.run_interactive(use_dashboard=True):
            context.input()
            renderer.step()
            context.display()
    else:
        last = None
        for _ in context.run_headless(a.steps):
            out = renderer.step(dt=1 / 60)
            last = out if out is not None else last
        data = np.asarray(last.data if hasattr(last, 'data') else last._data)
        print(f'{model!r}: output {data.shape}, finite {bool(np.isfinite(data).all())}, '
              f'radiance mean {data[..., 3].mean():.4f} range [{data[..., 3].min():.4f}, {data[..., 3].max():.4f}]')
    context.free()
    return 0


if __name__ == '__main__':
    sys.exit(main())
