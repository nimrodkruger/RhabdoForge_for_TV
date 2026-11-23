"""
TrilobiteVision left head as a native RF eye: scaffold schema, loader, IDs, frames, widths, assets.

Tests needing the registered calibration asset skip unless it is imported:
    python -m rhabdoforge.hardware import trilobite-left@left-room-20261105-scaled004 --from <TV root>
Tests needing OpenGL (marked 'gl') skip when no context can be created.
Run from the RhabdoForge root:  python -m pytest tests/hardware -q
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rhabdoforge.compound_eyes import Model
from rhabdoforge.hardware import (
    TV_TO_RF, AssetError, load_asset, match_rows, rf_sampling_frame, scaffold_from_rays_meta,
    tv_native_basis, tv_to_rf_points_um, tv_to_rf_vectors,
)
from rhabdoforge.hardware import assets as hw_assets

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'examples'))
sys.path.insert(0, str(ROOT / 'morphological_scaffolds' / 'trilobite_left'))
import trilobite_eye  # noqa: E402

SCAFFOLD = ROOT / 'assets' / 'trilobite_left_scaffold.npz'
NATIVE = ROOT / 'assets' / 'drosophila_scaffold.npz'
REF = 'trilobite-left@left-room-20261105-scaled004'
DEG = np.pi / 180


def _asset_or_skip():
    asset = load_asset(REF)
    if not asset.verify(roles=('geometry',), hashes=False)['ok']:
        pytest.skip(f'{REF} not imported into {asset.root}')
    return asset


@pytest.fixture(scope='module')
def meta():
    return _asset_or_skip().rays_meta()


@pytest.fixture(scope='module')
def model():
    return trilobite_eye.build_model('single', 1.0, scaffold=str(SCAFFOLD))


@pytest.fixture(scope='module')
def rows(model):
    with np.load(SCAFFOLD) as s:
        return match_rows(model.positions, s['positions'], scale=1e-6)


# Scaffold

def test_scaffold_has_exactly_the_native_keys_shapes_and_dtypes():
    with np.load(SCAFFOLD) as a, np.load(NATIVE) as b:
        assert sorted(a.files) == sorted(b.files) == ['directions', 'eye_id', 'positions']
        for k in b.files:
            assert a[k].dtype == b[k].dtype and a[k].ndim == b[k].ndim
            assert a[k].shape[1:] == b[k].shape[1:]
        assert len(a['positions']) == 130 and not a['eye_id'].any()
        assert np.allclose(np.linalg.norm(a['directions'], axis=1), 1.0, atol=1e-6)


def test_scaffold_regenerates_bit_for_bit_from_the_registered_asset():
    _asset_or_skip()
    from run import build_scaffold
    s = build_scaffold(REF)
    with np.load(SCAFFOLD) as a:
        for k in a.files:
            assert np.array_equal(a[k], s[k])


# Native loader

def test_unmodified_native_loader_builds_it_without_trilobite(model):
    assert not any(m == 'trilobite' or m.startswith('trilobite.') for m in sys.modules)
    plain = Model.from_file(str(SCAFFOLD))      # no preset at all: the loader alone accepts the file
    assert plain.shape == (130, 1) and model.shape == (130, 1)


def test_auto_derived_lattice_values_are_degenerate_and_the_preset_never_uses_them(model):
    plain = Model.from_file(str(SCAFFOLD))
    assert np.median(plain.buffer['aperture_um']) > 1000.0               # ~11 mm from viewpoint spacing
    auto_acc = np.degrees(np.asarray(plain.acceptance_angles_rest))      # follows those apertures
    assert not np.allclose(auto_acc, 1.0, rtol=0.5)                       # (5.7 deg with the default bundle)
    assert np.ptp(np.degrees(plain.interommatidial_tilt)) > 20.0          # meaningless lattice tilt
    assert np.allclose(model.buffer['aperture_um'], trilobite_eye.APERTURE_UM)
    assert np.allclose(model.interommatidial_tilt, 0.0)
    with np.load(SCAFFOLD) as s:
        ioa = trilobite_eye.measured_ioa_rad(s['directions'])
    assert np.allclose(model.interommatidial_angles, ioa)


def test_axis_lattice_is_isotropic_and_neighbours_are_equally_spaced(meta):
    L = meta['lenslets']
    c = np.array([l['centre_direction'] for l in L], float)
    v = np.array([l['viewpoint_mm'] for l in L], float)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    a, b = np.arctan2(c[:, 0], c[:, 2]), np.arctan2(c[:, 1], c[:, 2])
    A = np.c_[v[:, :2], np.ones(len(v))]
    J = np.array([np.linalg.lstsq(A, a, rcond=None)[0][:2], np.linalg.lstsq(A, b, rcond=None)[0][:2]])
    s = np.linalg.svd(J, compute_uv=False)
    assert s[0] / s[1] < 1.01                                    # isotropic
    cos = np.clip(c @ c.T, -1, 1)
    np.fill_diagonal(cos, -1)
    nn = np.degrees(np.arccos(cos.max(1)))
    assert nn.min() > 0.05 and nn.max() / nn.min() < 1.05        # not degenerate, uniform


# Stable IDs after canonical reordering

def test_rows_are_reordered_and_recovered_one_to_one(model, rows):
    assert not np.array_equal(rows, np.arange(130))              # construction did reorder
    assert np.array_equal(np.sort(rows), np.arange(130))


@pytest.mark.parametrize('pair', [('L+02-03', 'L-02+03'), ('L-03+01', 'L+03-01'), ('L+01+04', 'L-01-04')])
def test_asymmetric_ids_map_both_ways_to_their_own_viewpoint_and_axis(meta, model, rows, pair):
    L = {l['lenslet_id']: l for l in meta['lenslets']}
    ids = scaffold_from_rays_meta(meta)['lenslet_ids']
    pos_um = np.asarray(model.positions).reshape(-1, 3) / 1e-6
    fwd = np.asarray(model.directions).reshape(-1, 3)
    found = []
    for lid in pair:
        # forward: lenslet ID -> its RF row
        k = int(np.flatnonzero(ids[rows] == lid)[0])
        found.append(k)
        assert np.allclose(pos_um[k], tv_to_rf_points_um(L[lid]['viewpoint_mm']), atol=1e-2)
        assert np.allclose(fwd[k], tv_to_rf_vectors(L[lid]['centre_direction']), atol=1e-6)
        # reverse: RF row -> lenslet ID
        assert ids[rows[k]] == lid
    assert found[0] != found[1]
    # Mirror IDs are not mirror positions through the RF origin by accident of a sign flip
    assert not np.allclose(pos_um[found[0]], -pos_um[found[1]], atol=1.0)


# Frames and orientation

def test_frame_map_is_a_proper_rotation_and_flips_v_and_z():
    assert np.isclose(np.linalg.det(TV_TO_RF), 1.0)
    assert np.allclose(tv_to_rf_vectors([0, 0, 1]), [0, 0, -1])      # toward the scene -> RF forward
    assert np.allclose(tv_to_rf_vectors([0, 1, 0]), [0, -1, 0])      # native +v (image down) -> RF down


def test_rf_sampling_axes_follow_native_u_and_v_for_every_lenslet(meta, model, rows):
    L = sorted(meta['lenslets'], key=lambda l: l['index'])
    c_tv = np.array([L[r]['centre_direction'] for r in rows], float)
    e_u, e_v = tv_native_basis(c_tv)
    F = np.asarray(model.rhabdomere_directions).reshape(-1, 3)        # R = 1: receptor = lens axis
    T, B = rf_sampling_frame(F, np.asarray(model.buffer['right']).reshape(-1, 3))
    assert np.allclose(F, tv_to_rf_vectors(c_tv), atol=1e-6)
    # Same frame up to a signed rotation about F (RF right is perpendicular to world up; TV e_u is X
    # projected): measure it, never assume it is zero.
    mu, mv = tv_to_rf_vectors(e_u), tv_to_rf_vectors(e_v)
    angle = np.arctan2(np.sum(T * mv, axis=1), np.sum(T * mu, axis=1))
    assert np.abs(angle).max() < 2e-4                                 # rad (~0.01 deg)
    assert np.all(np.sum(B * mv, axis=1) > 0.9999)                    # B along native +v, not -v
    assert np.allclose(np.sum(T * F, axis=1), 0, atol=1e-6) and np.allclose(np.sum(B * F, axis=1), 0, atol=1e-6)


# Acceptance widths (shader convention)

def _shader_pure_importance(fwhm_min, fwhm_maj, n, seed=0):
    """numpy replica of sampling.glsl sampledir_importance at T=(1,0,0), B=(0,1,0), F=(0,0,1), tilt 0."""
    rng = np.random.default_rng(seed)
    u1, u2 = rng.random(n), rng.random(n)
    K = 4.0 * np.log(2.0)                                           # commons.glsl GAUSS_CONSTANT_K
    r = np.sqrt(-np.log(u1) / K)
    phi = 2 * np.pi * u2
    return np.tan(fwhm_min * r) * np.cos(phi), np.tan(fwhm_maj * r) * np.sin(phi)


@pytest.mark.parametrize('fwhm', [(1.0, 1.0), (1.0, 2.0)])
def test_half_maximum_falls_at_half_the_requested_width_on_each_axis(fwhm):
    fa, fb = fwhm[0] * DEG, fwhm[1] * DEG
    x, y = _shader_pure_importance(fa, fb, 4_000_000)
    for along, width, other in ((np.arctan(x), fa, np.arctan(y)), (np.arctan(y), fb, np.arctan(x))):
        strip = np.abs(other) < 0.05 * width                       # thin strip through the centre
        h, edges = np.histogram(np.abs(along[strip]), bins=60, range=(0, width))
        centres = (edges[1:] + edges[:-1]) / 2
        prof = h / h[:3].mean()
        half = np.interp(0.5, prof[::-1], centres[::-1])           # where the profile crosses 1/2
        assert half == pytest.approx(width / 2, rel=0.03)


def test_preset_acceptance_is_the_requested_full_width(model):
    acc = np.asarray(model.acceptance_angles_rest).reshape(-1, 2)
    assert np.allclose(acc, 1.0 * DEG)
    ell = trilobite_eye.build_model('single', (1.0, 2.0), scaffold=str(SCAFFOLD))
    acc = np.asarray(ell.acceptance_angles_current).reshape(-1, 2)
    assert np.allclose(acc[:, 0], 1.0 * DEG) and np.allclose(acc[:, 1], 2.0 * DEG)


def test_open7_receptors_stay_inside_the_lenslet_window():
    m = trilobite_eye.build_model('open7', 1.0, scaffold=str(SCAFFOLD))
    assert m.shape == (130, 7)
    axis = np.asarray(m.directions).reshape(-1, 1, 3)
    rd = np.asarray(m.rhabdomere_directions).reshape(-1, 7, 3)
    off = np.degrees(np.arccos(np.clip(np.sum(rd * axis, -1), -1, 1)))
    assert off[:, 6].max() < 0.1                                    # centre slot on axis
    ring = off[:, :6]
    assert 1.0 < ring.mean() < 2.0
    # receptor offset + kernel support (2 FWHM) inside the +-5.3 deg lenslet window
    assert ring.max() + 2 * 1.0 < 5.3


# Calibration asset registry

def _fake_tv_tree(root: Path):
    b = root / 'analysis/run/calibration/left'
    b.mkdir(parents=True)
    (b / 'rays.json').write_text(json.dumps({
        'model': {'model': 'focused_plenoptic_virtual_camera_array'}, 'trial': 't', 'camera': 'left',
        'status': 'test', 'frame': {'units': 'mm'}, 'lenslets': [{'index': 0}]}))
    np.save(b / 'rays.npy', np.zeros((2, 2, 6), np.float32))
    np.save(b / 'lenslet.npy', np.zeros((2, 2), np.int16))
    run = root / 'analysis/run'
    (run / 'run.json').write_text('{}')
    ref = run / 'references/rec-1/left'
    ref.mkdir(parents=True)
    for name in hw_assets.REFERENCE_FILES:
        (ref / name).write_bytes(b'x' * 8)
    lat = run / 'lattice/rec-1/left'
    lat.mkdir(parents=True)
    (lat / 'roi.json').write_text('{}')
    rec = root / 'data/rec_src'
    rec.mkdir(parents=True)
    (rec / 'recording.json').write_text('{}')


def _register(tmp_path, version):
    src = tmp_path / 'tv'
    if not src.exists():
        _fake_tv_tree(src)
    m = hw_assets.build_manifest(
        asset='fake-left', version=version, label='l', status='s', source_root=src,
        bundle='analysis/run/calibration/left', calibration_run='analysis/run', session_id='rec-1',
        head='left', source_recording='data/rec_src', provenance={})
    hw_assets.write_manifest(m, tmp_path / 'manifests')
    return src


def test_import_verify_and_corruption_are_reported(tmp_path):
    src = _register(tmp_path, 'v1')
    asset = load_asset('fake-left@v1', manifest_dir=tmp_path / 'manifests', store=tmp_path / 'store')
    assert not asset.verify()['ok']                                 # nothing imported yet
    with pytest.raises(AssetError, match='import'):
        asset.require('geometry')
    assert asset.import_from(src)['ok']
    assert {e['role'] for e in asset.files()} == {'geometry', 'photometry', 'source_record'}
    p = asset.path('calibration/rays.npy')
    data = bytearray(p.read_bytes())
    data[-1] ^= 0xFF
    p.write_bytes(bytes(data))
    report = asset.verify()
    assert not report['ok'] and report['problems'] == [{'path': 'calibration/rays.npy', 'problem': 'sha256'}]
    assert asset.verify(hashes=False)['ok']                         # size alone cannot see it


def test_bare_name_resolves_only_when_one_version_exists(tmp_path):
    _register(tmp_path, 'v1')
    assert load_asset('fake-left', manifest_dir=tmp_path / 'manifests').ref == 'fake-left@v1'
    _register(tmp_path, 'v2')
    with pytest.raises(AssetError, match='2 registered versions'):
        load_asset('fake-left', manifest_dir=tmp_path / 'manifests')
    with pytest.raises(AssetError, match='unknown'):
        load_asset('fake-left@v3', manifest_dir=tmp_path / 'manifests')


def test_registered_manifest_lists_what_replay_needs_and_is_found_from_any_directory(tmp_path):
    m = load_asset(REF).manifest
    files = {e['path'] for e in m['files']}
    for name in ('rays.json', 'rays.npy', 'lenslet.npy'):
        assert f'calibration/{name}' in files
    for name in ('descriptor.json', 'dark_mean.npy', 'pix_std.npy', 'hot.npy'):
        assert f'run/references/rec-192719/left/{name}' in files
    assert 'run/lattice/rec-192719/left/roi.json' in files and 'run/run.json' in files
    assert m['hardware_tag'] is None and m['hardware_tag_assignment']['assigned_by'] is None
    assert m['calibration_digest'] == hw_assets.digest_of([e for e in m['files'] if e['role'] == 'geometry'])
    out = subprocess.run([sys.executable, '-m', 'rhabdoforge.hardware', 'where', REF],
                         cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    where = json.loads(out.stdout)
    assert where['session_id'] == 'rec-192719' and where['preprocessing'] == 'dark_only'


# Rendering (OpenGL)

@pytest.fixture(scope='module')
def gl_context():
    try:
        from rhabdoforge.engine import Context
        ctx = Context()
    except Exception as e:                                          # pragma: no cover
        pytest.skip(f'no OpenGL context: {e}')
    yield ctx
    ctx.free()


@pytest.mark.gl
def test_viewpoint_parallax_splits_the_array_at_a_near_edge(gl_context):
    """A crate edge on x = 0, 0.7 m ahead: viewpoints right of it see the face, left of it the sky."""
    from rhabdoforge.engine import Agent, Asset, Scene
    from rhabdoforge.engine.meshes import CUBE_INDICES, CUBE_VERTICES
    from rhabdoforge.renderers import Renderer

    scene = Scene(background_color=(0.0, 0.0, 0.0))
    scene.add_sky('assets/textures/kloppenheim_05_4k.exr')
    pos, uvs = np.split(CUBE_VERTICES.reshape(-1, 5), [3], axis=1)
    half = float(pos.max(0)[0])
    crate = Asset.from_arrays(name='crate', vertices=pos, faces=CUBE_INDICES.reshape(-1, 3), uv_coords=uvs,
                              texture='assets/textures/wood.jpg', sRGB=False)
    scene.add_instance(asset=crate, transform=(half, 0.0, -0.7 - half))
    m = trilobite_eye.build_model('single', 1.0, scaffold=str(SCAFFOLD))
    r = Renderer(model=m, scene=scene, agent=Agent(position=(0, 0, 0)), nb_samples=256,
                 time_dithering=False, enable_microsaccades=False)
    out = None
    for _ in gl_context.run_headless(4):
        out = r.step(dt=1 / 60) or out
    raw = np.asarray(out.raw_radiance).reshape(-1)
    x_mm = np.asarray(m.positions).reshape(-1, 3)[:, 0] * 1e3
    sky, face = raw[x_mm < -20].mean(), raw[x_mm > 20].mean()
    assert np.isfinite(raw).all() and sky > 0.05 and face < 0.2 * sky
